---
title: A Good Catan Player That Is Not Me! 
category: RL
tags: [RL]
created_at: 2026-09-26
updated_at: 2025-09-26
language: en
---

# A Good Catan Player That Is Not Me! 

The pure unbridled joy of winning a Catan game is not something a learned policy can understand, but through reward signals the mechanism can get pretty close! I set out Bear Grylls style to find a wild set of weights and tame it through RL into becoming a Catan prowess that can trigger all my friends mwahaha. But to get to a similar Catan bot vs human showdown as 2016's Lee Sedol vs AlphaGo, the journey was a bit like: two roads diverged into a yellow wood, and to know which to follow, I would try out both.      

### What makes Catan hard to learn

Naturally one might ask why hasn't every board game been solved by RL by now? Given superhuman performances in games like Chess, Go and even StarCraft, it would seem that sigificant progress has been made against many types of game mechanics. One constraint is that RL aims to approximate optimal strategy, but does not provide any sort of guarantee of correctness. It learns to estimate from games it has played how good each move is, then shifts toward the moves that did better than expected, but its ability to learn comes down to how noisy those estimates are.  

In chess and Go, the same moves from the same position always produce the same game, and both players see the whole board. Catan adds four sources of noise:
1. Dice rolls determine resource production
2. Dev cards are drawn randomly from a fixed distribution and player hands are hidden from opponents (which may be uncertain after a knight or robber is played)
3. Up to four players compete for the same resources and board space
4. Trades rely on another player to go through

On top of that, the reward is assigned at the end: +1 for the winner, −1/(n-1) for the remaining n players. This means a player shares credit for the result across its several hundred game decisions.  

So the project is mostly about two things: making these estimates less noisy and measuring whether game play improved.

### Setup

The version of Catan played is the one seen on Colonist.io. For 1v1: the first to 15 VPs wins, the discard limit is 9 when a 7 is rolled, the friendly robber is on (the robber can't be put on a hex touching a player with 2 or fewer visible points, and that player can't be robbed), dice are balanced, and there is no player-to-player trading. For 4 players: the first to 10 VPs wins, the discard limit is 7, the friendly robber and balanced dice are on, and players can trade with each other, up to 4 cards on each side of a deal.

RL needs three things from a game: what the agent sees, what it can do, and what it's rewarded for. 

### What the network sees
At each decision point, the network recieves a fixed-size, 369-byte snapshot `Observation` of the game. The mover is always represented as player 0 (P0) to keep the representation consistent. The snapshot includes the board, every building (packed as bitmasks), the mover's hand, the bank, what phase the turn is in, and any trades made during the turn. An example from a 2-player game is shown below.

<iframe src="/embeds/catan/state.html" title="Catan state" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 850px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Raw bytes are effectively unlearnable, and bitmasks reside in a special pineapple at the bottom of the sea. The network's forward function addresses this limitation of our representation by using arithmetic (nothing learned) to unpack the 369-byte snapshot into six matrices arranged by board piece: $19 \times 29$ for the 19 hexes, $54 \times 33$ for the 54 corners, $72 \times 16$ for the 72 edges, $4 \times 48$ for the players, $5 \times 27$ for the five resources, and $1 \times 203$ for the remaining game information. An example for a four-player unpacked representation is shown below.

<iframe src="/embeds/catan/decode.html" title="Catan decode" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 720px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

However, these matrices have different column counts (29, 33, 16, 48, 27, 203). We want to combine them because the information that matters for a decision is spread across them:
- Whether a corner is a good settlement spot depends on its hexes
- Whose roads are closing in, and which player is ahead and has the resources to race you there
- Game information like the phase and an ongoing trade lives in the summary

They cannot be stacked into one grid yet. So, each matrix is passed through its own learned layer which turns each row into 128 numbers. For instance, a hex's 29 numbers (in the $19 \times 29$ matrix) will become 128 numbers through a $29 \times 128$ learned matrix plus 128 biases, and all 19 rows use that same matrix. Corners, edges, players, resources and the summary each have their own. Every row is now 128 wide, so they stack into one grid: summary (1), players (4), resources (5), hexes (19), corners (54), and edges (72), giving $155 \times 128$.

<iframe src="/embeds/catan/embed.html" title="Catan embed" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 480px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

After the per-kind matrices, each row is 128 numbers describing what's at that spot. For example: an empty corner touching wheat 6 and ore 8, with nobody's road nearby. But the row does not store where it is on the board. This is an issue because later the transformer will treat the 155 rows as position agnostic. If two corners on opposite sides of the board have identical features, their rows are identical and the network will not be able to distinguish them.

To address this, each row gets a learned offset that encodes its position which start small. The row mostly still describes what is there, nudged slightly by where it is, so a coastal corner's position offset can come to mean "on the coast". The summary and the 145 board rows share a $146 \times 128$ position-offset matrix, and the players get their own $4 \times 128$ seat offsets. Resources need no offset: each resource row already says which resource it is in its own features.

A small aside, why would we not just give each corner an ID to mark its position? We could append 54 extra columns to every corner's features: all 0, except a 1 in that corner's own slot. Each corner's input becomes 33 + 54 = 87 numbers and its learned matrix grows from $33 \times 128$ to $87 \times 128$. 

But if you look at what those extra 54 columns do, in multiplication the single 1 gets multiplied by one row of the matrix, which really means one learned vector is added per corner. This is the same as the position offset (without multiplying/storing extra 0s).

The $155 \times 128$ grid then passes through 4 transformer blocks. Each returns a new $155 \times 128$ grid, and each has two steps: attention and a feed-forward network. Before each step, every row is rescaled to a standard size (a layer norm), which keeps the numbers from drifting as they pass through the blocks.

The first step of a transformer block is attention. Each row wants to collect information from some of the other 154 rows, weighting the ones that matter to it. A corner cares about its hexes, but perhaps not about an edge on the far side of the board. So each row needs to answer two questions:
- Which rows should I listen to?
- What should I take from them?

One approach is to score each pair of rows by how similar they are, then take the weighted average. This works poorly because a corner needs its hexes, which look nothing like a corner, but similarity would point it at other corners instead. Similarity is also symmetric and always highest with itself, so a corner would care about a hex exactly as much as the hex cares about it, and every row would mostly listen to itself. Then, each row would pass along everything it has, rather than just the parts worth sharing.

Instead, for each row of the $155 \times 128$ matrix, the network computes three 128-long vectors using three learned matrices (stored as three $128 \times 128$ matrices stacked: a $128 \times 384$ grid): a query (what this row searches for), a key (how other rows find it), and a value (what it passes on when found). Each 128-long query, key and value is cut into 4 pieces of 32-long length. Piece 1 of everything belongs to head 1, piece 2 to head 2, and so on. Each head has its own Q, K and V, each $155 \times 32$.

<iframe src="/embeds/catan/qkv-split.html" title="Catan attention split" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 240px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Within each head, every row's query is multiplied with every row's key and the products are added up (then divided by $\sqrt{32}$ to keep the numbers tame). If the two are large in the same places, the sum is large, so it's a good match. This gives each row 155 scores, one for every row, including itself. A learned bonus is added to each score depending on how the two rows are related: a hex and its corner, two neighbouring corners, an edge and its corner, two touching edges, a row and itself, the summary and anything, a player and a resource, and so on, 12 relations in all, with "unrelated" for the rest. Each block and each head learns its own bonuses, and "unrelated" starts at −2, so far-off rows begin mostly ignored. A softmax then turns the 155 scores into 155 percentages that add up to 100%. The head's output is the sum of all 155 rows' values, each weighted by its percentage. For example, a corner might learn to put most of its weight on its own hexes.

<iframe src="/embeds/catan/qkv-head.html" title="Catan attention head" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 460px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

The four heads run these searches in parallel, each with its own 32-number slice. Their four 32-number results are placed side by side to make 128 numbers, passed through one more learned $128 \times 128$ matrix, and added to the row. All 155 rows share the same matrices within a transformer block, and each of the 4 blocks has its own. They all start random.

<iframe src="/embeds/catan/qkv-combine.html" title="Catan attention heads" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 265px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

The second step in a transformer block is a feed forward network. Each row passes through a small two layer network: its 128 numbers turn into 512, pass through a GELU (which squashes negative numbers to about zero and keeps positive ones) and turn back into 128. The result is added to the row. All 155 rows use the same two matrices.

<iframe src="/embeds/catan/ff.html" title="Catan feed forward" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 345px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

To recap, attention gives each row a weighted average of the values of the rows it listened to. The feed forward network then makes sense of what was just gathered. For example, one of the 512 columns of the first $128 \times 512$ matrix might weight "near ore" positively, "near grain" positively, and "opponent road nearby" negatively. Its number comes out high when the row has ore and grain around it and no opponent road, so it acts as a learned question. The 512 numbers are answers to 512 such questions. The GELU lets the strong yeses through and squashes the no's to about zero. The second matrix ($512 \times 128$) turns those answers into a delta added to the original row. If the "strong city spot" question fired, the row gets nudged in a direction that means "strong city spot", which later blocks can search for through attention.

Both steps add their result to the row instead of replacing it which is called a residual connection. It does this so it only has to learn what to change and nothing the row already knew is lost. The addition also makes the network trainable. During backprop, every row gets a message for the weights that produced it, something like "this number should have been a bit higher," or lower. Because each step adds its result onto the row instead of replacing it, that message reaches the earlier blocks just as strongly as the later ones, instead of fading as it passes back through each matrix.

## What the network can do

After the last block, every row of the $155 \times 128$ grid has been updated four times. The network will turn the grid into a move.

This must be one of 257 moves: 54 settlement spots, 54 city upgrades, 72 roads, 19 robber hexes, 7 turn moves like rolling, and the rest for naming resources, trading and robbing.

<iframe src="/embeds/catan/moves.html" title="Catan moves" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 475px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Choosing directly among all 257 moves would treat "build a road" as 72 unrelated moves and learning that a road was the right call would only teach the one edge that was played. So the network decides in two steps, like OpenAI Five and AlphaStar: first what kind of move (one of 25), then, if needed, where or with whom.

The kind of move is a whole-game decision, so it is read from the summary row (row 0), the one row not tied to any piece. In every block it can attend to all 155 rows, and because the kind of move is read from it, training teaches it to gather whatever that choice needs. Row 0 is multiplied by a learned $128 \times 25$ matrix, which gives 25 numbers, one per kind of move. 

A softmax turns these 25 scores into percentages which add up to 100%. Most kinds are illegal at any moment, and the network shouldn't have to learn the rules to avoid them. So with each 369-byte snapshot the engine sends a list of the legal moves, and before the softmax every illegal kind's number is set to $-\infty$. Since $e^{-\infty} = 0$, illegal kind of moves get 0%, can never be picked, and receive no gradient in training. The same is done for targets of the moves in the second step. The chance of a move is then the chance of its kind of move times the chance of its target.

<iframe src="/embeds/catan/pick-kind.html" title="Catan move kind" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 270px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Once the kind of move is chosen, say "build a road", the network still has to pick which edge. Every edge already has its own row in the post-transformer blocks grid, and after four blocks that row describes the edge in context: its corners, its hexes, and whose roads are nearby. So picking an edge comes down to asking which edge's row best matches what the network wants right now. That is the same question attention asks, so the network answers it the same way: the summary row produces a query (what a good road spot looks like at this moment), each legal edge's row produces a key (what that edge offers), and the edge whose key best matches the query scores highest.

In numbers: the summary row (row 0) times a learned $128 \times 128$ matrix gives the road query. Each kind of move has its own query matrix, so the summary row asks a different question for roads than for settlements. Each edge's row times a second learned $128 \times 128$ matrix gives that edge's key. This key matrix is shared by every kind, so an edge describes itself the same way whatever is being asked. An edge's score is its key dotted with the query, divided by $\sqrt{128}$ to keep the numbers small: it is high when the edge has what the query is looking for. As with the kinds, illegal edges are set to $-\infty$, and a softmax turns the rest into chances. Settlements and cities work the same way over the corner rows, the robber over the hex rows, and robbing over the player rows.

<iframe src="/embeds/catan/pick-target.html" title="Catan move target" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 430px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Trade offers in a four-player game can't be chosen this way, because there's no fixed list of deals to point at. A deal can be almost any mix of cards to give and cards to ask for. So once the network chooses "offer", it builds the deal one card at a time. Each round it either stops, or adds one card: one more of some resource to give, or one more to ask for.

Each choice depends on what is already in the deal. So before every round the network describes the deal so far as 128 numbers. It starts from the summary row times a learned $128 \times 128$ matrix. There are two of these learned matrices, one for proposing an offer and one for countering someone else's, since a good card to add isn't the same in both. Then, for each of the five resources, it adds a learned vector for that resource's current count. Each resource has one vector per count, from "give 12" to "ask 12", so "give 2 brick, ask 1 ore" is the brick vector for −2, plus the ore vector for +1, plus the zero-count vectors for the other three.

These 128 numbers are then turned into 11 scores. A small network reads them and gives one score for stopping. For the cards, the 128 numbers are placed next to one resource's row of the grid, and a second small network reads both and gives two scores, "give one more of this resource" and "ask for one more". The same network runs once for each of the five resources, with only the resource's row changing, which makes 10 card scores. Cards that break the rules (giving what you don't hold, more than four a side, a deal already offered) are set to $-\infty$, and a softmax over the rest picks the next step.

<iframe src="/embeds/catan/pick-offer.html" title="Catan offer" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 270px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Each other player then accepts, declines or counters, and the proposer trades with an accepter, takes a counter-offer, or cancels. A turn has a budget of decisions after the roll, which keeps a player from offering forever: 15 in 1v1, and 8 in the current 4-player run, where every offer also draws an answer from each of the other three players.

## What the network is rewarded for

When a game ends, the winner gets $+1$ and every other player gets $-1/(n-1)$, so the rewards in each game add up to zero. One player's win is everyone else's loss, a Catan story as old as time. It is a difficult task to issue updates to the players since a game is about a thousand decisions long and all training gets is this one number at the end.

The obvious way to learn from it is to make every move of a won game a bit more likely and every move of a lost game a bit less likely. This is the policy gradient, basically nudge the weights $\theta$ along $\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R$, where $\pi_\theta(a \mid s)$ is the probability the network gives move $a$ in position $s$ and $R$ is the reward. On average it points the right way, but Catan play is mostly dice. Even a strong player loses three four-player games in four, so good moves get punished all the time, and most of a result comes from the position a player was in rather than from any one move.

So instead, each move is judged against what was expected. The network has a second output, the forecast: from any position, its guess of the reward each player will end up with, between $-1$ and $+1$. (It multiplies the same summary row by a learned $128 \times 4$ matrix and squashes the result with $\tanh$.) A move's credit, its advantage, is after minus before. That is, how things went following the move, minus the forecast when it was made. A move that keeps a strong position strong earns about nothing, but a move that turns a game around earns a lot.

Each change in the forecast is a surprise, $\delta_t = V(s_{t+1}) - V(s_t)$, with the real result in place of the next forecast once the game ends. Surprises right after the move are mostly its doing; ones fifty decisions later mostly aren't. So each one counts a factor of $\lambda = 0.95$ less than the one before it:

$$A_t = \delta_t + \lambda\,\delta_{t+1} + \lambda^2\,\delta_{t+2} + \cdots$$

<iframe src="/embeds/catan/credit.html" title="Catan credit" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 385px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

There is a catch here though. A move's credit depends on what happens after it, so the network has to play first and look back later. A game is about a thousand decisions long, which is too long to wait, so it looks back every 64 decisions. If a game ended in that stretch, *after* runs all the way to the real result. If it didn't, the latest forecast stands in for the result that hasn't happened yet.

That makes every credit only as good as the forecast, so the forecast has to learn too, and the same subtraction tells it what to learn. If the credit is *after* minus *before*, then *after* is *before* plus the credit: what the forecast should have said, $y_t = V(s_t) + A_t$. The forecast is pulled toward it, with the loss $(V_\theta(s_t) - y_t)^2$. Most of *after* is later forecasts, which could just chase each other, but the end of every game is real, so the last forecasts are pulled toward real results, the ones before them toward those, and over time that accuracy spreads back to the start of the game.

Now every decision has a credit, and every forecast has a target. A single credit is still mostly luck, though, so the network never learns from one decision at a time. With 2,048 games running at once, every 64 decisions it has 131,072 recorded decisions. It shuffles them and deals them into 32 groups of 4,096. Each group changes every weight once: each decision's change is worked out, and the group's 4,096 changes are averaged into one. Changes caused by luck point in different directions and cancel, while changes caused by decisions that are reliably good or bad point the same way and survive. Before averaging, the credits in a group are rescaled to average 0 and spread 1, so each decision is judged against the others in its group.

<iframe src="/embeds/catan/loop.html" title="Catan training loop" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 232px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Let us discuss further what a move's change actually is. The network chose its kind of move from 25 types of moves, turned into chances by a softmax, so the credit first decides how each score should move:

$$\Delta z_k = \begin{cases} A_t\,(1 - \pi_k) & \text{for the kind of move that was played,} \\ -A_t\,\pi_k & \text{for every other kind.} \end{cases}$$

Here $z_k$ is the score of kind of move $k$, $\pi_k$ its chance, and $\Delta z_k$ how much its score should change.

Basically, if the move went well, its score goes up by the credit times the chance it was not taken, so a move that was already 96% likely barely changes. Every other allowed move's score goes down by the credit times its own chance, so the strongest alternatives lose the most. Moves that weren't allowed don't change, and if the move went badly, everything reverses.

<iframe src="/embeds/catan/backprop.html?frame=scores" title="Catan backprop: scores" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 205px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

The network doesn't store scores, though. It calculates them by multiplying numbers through its weights, so the only way to change a score is to change weights. Backpropagation decides how, starting at the scores and working back one matrix at a time. Each weight changes by the number it multiplied times how much its result should change: a weight that multiplied a large number had a large effect, so it changes a lot, and one that multiplied zero doesn't change. Each matrix then works out how its own input should have been different and hands that back to the matrix before it, which does the same.

<iframe src="/embeds/catan/backprop.html" title="Catan backprop" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 480px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

In each block, the feed forward's answers that were off for this position played no part, so they get no changes; each block added its result on top of the row it received, so the row it recieved is still part of what it passed on and any change asked of the block's output is asked of its input (so it passes the requested change to the block below as well); and row 0's attention output was a mix of other rows, so the change is shared out to them, more to the ones it drew from most.

There is only one training process. After each move, two things at the top of the network are wrong: the scores for each kind of move, which should shift by the credit, and the forecast, which should move toward its target. Both errors are sent back down through the same network in the same backward pass, and wherever they meet they simply add up. So each minibatch makes a single change to the weights, and that one change improves the move choices and the forecast together. It is also why the forecast makes the moves better. Both read from the same blocks, so anything the blocks learn about where a game is heading is already there when the network scores its moves.

Every position runs through the same weights, so a change made for one position doesn't stay there. Any position with similar features, a corner touching a 6 and an 8, or an opponent's road closing in, shifts too. The network never sees the same position twice, so this is the only way it could get better: by learning patterns that carry over to positions it hasn't seen.

Playing is the slow part, so the network gets two uses out of every recorded decision. After all 32 groups have changed the weights, it shuffles the same 131,072 decisions into 32 new groups and goes through them again, 64 changes in all, before it plays again. But after a few changes, the network being trained is no longer the same one that made those original decisions. The further a move's chance drifts from what it was when played, the less its credit applies. Proximal Policy Optimization (PPO) aims to restrict this drift. With $r_t = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$, the move's chance now divided by its chance when it was played,

$$L_{\text{policy}} = -\min\big(r_t A_t,\ \operatorname{clip}(r_t, 0.8, 1.2)\,A_t\big),$$

so once a move is 20% more likely than when it was played (or 20% less likely, for a bad one), it stops being pushed. Then the updated network plays the next 64 decisions of the same games and the loop repeats. One round of playing, looking back and learning is an iteration.

Left to run for thousands of iterations, the loop can go wrong in two ways. First, a network that only reinforces what worked tends to pile all its probability onto one move early and stop trying anything else. So the loss also rewards keeping its chances spread out, measured by their entropy $H$ (one for the kind of move and one for its target). That reward is weighted by $\beta$, which falls from 0.01 to 0.001 as the network improves and exploring matters less. Second, nothing so far makes the summary row remember what the player can't see, like the cards in each opponent's hand. So two extra outputs guess each opponent's hidden cards and how each would answer a trade offer, graded against the true answers, which training knows. The guesses are never used in play, but making them well forces the summary row to keep count. All together:

$$L = L_{\text{policy}} + L_{\text{value}} - \beta\,(H_{\text{kind}} + H_{\text{target}}) + 0.1\,L_{\text{hands}} + 0.1\,L_{\text{answers}}.$$

Three smaller choices help it get going and stay steady. A brand-new network almost never wins, so in the four-player run each player also gets 0.03 for every victory point it gains relative to the others, fading to zero over the first 2,000 iterations. When it learns, every recorded position is turned by one of the board's 12 symmetries (six rotations, each with or without a mirror), chosen at random each time, so a lesson learned in one corner of the board holds in all of them. And if a minibatch asks for an unusually large change, the change is scaled down to a fixed maximum size, so one odd minibatch can't throw the weights far.

## Roads trodded on

Everything here was trained on a single 24 GB 4090 GPU using Vast. The network is small, four blocks 128 wide, about a million weights. One iteration, 2,048 games advancing 64 decisions each, takes about 20 seconds: around 4 to play and 15 to learn. The final 1v1 network trained for 13,398 iterations, 1.76 billion decisions in about 44 hours. A four-player run from scratch went 9,173 iterations, 1.2 billion decisions in about 24 hours; the four-player network in the player below is the strongest checkpoint of a second run, from iteration 4,000. Both share the settings above (a learning rate of $3 \times 10^{-4}$, minibatches of 4,096, two passes, PPO's 20% limit, entropy weight falling from 0.01 to 0.001). The final 1v1 run adds three things: player-to-player trading is off, the forecast is trained with a critic that sees every hand, and in 20% of its games it plays one of four earlier versions of itself, refreshed every 250 iterations, so it can't get good only at beating its current self.

Self-play can't measure its own progress: a network playing copies of itself wins its fair share of games no matter how good it is. So progress is measured against a fixed opponent instead, a hand-written bot that builds the most valuable thing it can afford, trades with the bank only to afford something, accepts an offer only if it lets it build, and robs the leader. It is never learned from. In these runs the network takes half the seats and the bot the other half, so an even match is 50%.

<iframe src="/embeds/catan/learning.html" title="Catan learning curves" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 525px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Both a four-player and a 1v1 run from scratch pass even within about 50 iterations and win around 90% of games against the bot by 10,000. The other panels show what changed in how they play. Trading starts out frantic, over 100 trades per 1,000 decisions, because random offers cost nothing to make, and falls as the network learns that most of them lead nowhere: the four-player run settles near 21. The 1v1 run almost stops trading altogether. With only one other player, every trade the opponent accepts helps the opponent too, and the final 1v1 run leaves trading between players out. In the final 1v1 run, the opening settlements climb from about 9.7 pips to over 11 in the first 60 iterations, then settle near 10.9, and games shorten from about 300 decisions to around 200.

Before those long runs, a handful of ideas were each tried in a short four-player run of about 300 iterations, against the same setup without them. In these runs one network plays three bots, so an even match is 25%.

<iframe src="/embeds/catan/ideas.html" title="Catan what was tried" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 215px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Two of them tried to take some of the luck out of each move's credit. One trained the forecast with a critic that can see every player's hand, so it knows what each player can actually build. The other judged a move followed by a dice roll against the average over all eleven possible rolls, rather than the one that came up. A third showed every position under a random rotation or reflection of the board, to teach each lesson twelve times over. A fourth trained the network toward the moves a search prefers, as in Gumbel AlphaZero, which made every iteration nearly twice as slow. In a short run, each of them slowed learning down: none reached even, and head to head against the plain run, the symmetry run won 2.3% of its games and the critic run 5.4%. Over longer runs they can still pay off, and symmetry and the critic both ended up in the final 1v1 run.

Searching ahead does help at play time: a player that searches before every move wins 38.5% of four-player games against the network alone. The player below doesn't search; it runs the network on its own, in the browser.

## Play the network
<iframe src="/embeds/catan/play.html" title="Play the network" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 640px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

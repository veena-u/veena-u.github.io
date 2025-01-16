import React, { useState, useEffect, useRef } from 'react';
import type { FC } from 'react';
interface Position {
  x: number;
  y: number;
}

const NekoComponent: FC = () => {
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const [mousePos, setMousePos] = useState<Position>({ x: 0, y: 0 });
  const [sprite, setSprite] = useState<string>('awake');
  const [frame, setFrame] = useState<number>(1);
  const [isWaiting, setIsWaiting] = useState<boolean>(false);
  const [mounted, setMounted] = useState<boolean>(false);
  const [idleState, setIdleState] = useState<number>(0);
  const [stateCount, setStateCount] = useState<number>(0);
  const [isHidden, setIsHidden] = useState<boolean>(false);
  const animationFrameRef = useRef<number | undefined>(undefined);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const isMovingRef = useRef<boolean>(false);
  const isBoinkedRef = useRef<boolean>(false);
  const [isWashing, setIsWashing] = useState<boolean>(false);

  // Constants
  const SPEED = 2;
  const FRAME_INTERVAL = 200;
  const CAT_SIZE = 40;
  const STATE_DURATION = 30; 

  const IDLE_STATES = [
    { maxState: 2, sprite: 'awake' },
    { maxState: 5, sprite: 'scratch' },
    { maxState: 8, sprite: 'wash' },
    { maxState: 11, sprite: 'yawn' },
    { maxState: Infinity, sprite: 'sleep' }
  ];

  const getSpriteName = (baseName: string, frame: number): string => {
    return baseName === 'awake' ? 'awake' : `${baseName}${frame}`;
  };

  const handleIdle = () => {
    setStateCount(prev => prev + 1);

    if (stateCount >= STATE_DURATION) {
      setStateCount(0);
      setIdleState(prev => prev + 1);
    }

    // Find the appropriate sprite for current idle state
    const currentState = IDLE_STATES.find(state => idleState <= state.maxState);
    setSprite(currentState?.sprite || 'sleep');
  };

  useEffect(() => {
    setMounted(true);
    if (typeof window !== 'undefined') {
      const pick = () => ({
        x: CAT_SIZE + Math.random() * (window.innerWidth - 2 * CAT_SIZE),
        y: CAT_SIZE + Math.random() * (window.innerHeight - 2 * CAT_SIZE)
      });
      const overIframe = ({ x, y }: Position) =>
        [[0, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]].some(([dx, dy]) =>
          document.elementFromPoint(x + dx * CAT_SIZE / 2, y + dy * CAT_SIZE / 2)?.closest('iframe')
        );

      let spot = pick();
      for (let tries = 0; tries < 50 && overIframe(spot); tries++) spot = pick();

      const start = { x: window.scrollX + spot.x, y: window.scrollY + spot.y };
      setPosition(start);
      setMousePos(start);
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;

    let clientPos: Position | null = null;

    const handleMouseMove = (e: MouseEvent) => {
      clientPos = { x: e.clientX, y: e.clientY };
      setMousePos({ x: e.pageX, y: e.pageY });
    };

    const handleScroll = () => {
      if (!clientPos) return;
      setMousePos({
        x: clientPos.x + window.scrollX,
        y: clientPos.y + window.scrollY
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('scroll', handleScroll);
    };
  }, [mounted]);

  useEffect(() => {
    if (!mounted) return;

    const handleClick = () => {
      setIsWaiting(!isWaiting);
      setIdleState(0);
      setStateCount(0);
    };

    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [isWaiting, mounted]);

  useEffect(() => {
    if (!mounted || isHidden) return;

    const animate =(currentTime: number) => {
      if (!lastTimeRef.current) lastTimeRef.current = currentTime;
      const deltaTime = currentTime - lastTimeRef.current;

      if (deltaTime > FRAME_INTERVAL) {
        frameCountRef.current = (frameCountRef.current + 1) % 2;
        setFrame(frameCountRef.current + 1);
        lastTimeRef.current = currentTime;
      }

      if (!isWaiting) {
        const dx = mousePos.x - position.x;
        const dy = mousePos.y - position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > CAT_SIZE) {
          isMovingRef.current = true;
          const angle = Math.atan2(dy, dx);
          const newX = position.x + Math.cos(angle) * SPEED;
          const newY = position.y + Math.sin(angle) * SPEED;

          if (idleState !== 0) {
            setIdleState(0);
            setStateCount(0);
          }

          const degrees = (angle * 180) / Math.PI;
          let newSprite = 'right';
          
          if (degrees > -22.5 && degrees <= 22.5) newSprite = 'right';
          else if (degrees > 22.5 && degrees <= 67.5) newSprite = 'downright';
          else if (degrees > 67.5 && degrees <= 112.5) newSprite = 'down';
          else if (degrees > 112.5 && degrees <= 157.5) newSprite = 'downleft';
          else if (degrees > 157.5 || degrees <= -157.5) newSprite = 'left';
          else if (degrees > -157.5 && degrees <= -112.5) newSprite = 'upleft';
          else if (degrees > -112.5 && degrees <= -67.5) newSprite = 'up';
          else if (degrees > -67.5 && degrees <= -22.5) newSprite = 'upright';

          setSprite(newSprite);
          setPosition({ x: newX, y: newY });
        } else {
          if (isMovingRef.current) {
            setIdleState(0);
            setStateCount(0);
            isMovingRef.current = false;
          }
          handleIdle();
        }
      } else {
        setSprite('sleep');
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [mousePos, position, isWaiting, mounted, idleState, stateCount, isHidden]);

  if (!mounted || isHidden) return null;

  const currentSpriteName = getSpriteName(isWashing ? 'wash' : sprite, frame);

  return (
    <div
      className="absolute cursor-pointer"
      onClick={(e) => {
        e.stopPropagation();
        if (isBoinkedRef.current) {
          setIsHidden(true);
          return;
        }
        isBoinkedRef.current = true;
        setIsWashing(true);
        setTimeout(() => setIsWashing(false), 800);
      }}
      style={{
        left: `${position.x - CAT_SIZE/2}px`,
        top: `${position.y - CAT_SIZE/2}px`,
        width: `${CAT_SIZE}px`,
        height: `${CAT_SIZE}px`,
        zIndex: 9999,
      }}
    >
      <img
        src={`/assets/neko/${currentSpriteName}.png`}
        alt="Neko"
        className="w-full h-full"
      />
    </div>
  );
};

export default NekoComponent;
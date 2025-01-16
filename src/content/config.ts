import { z, defineCollection } from "astro:content";
const blogCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    category: z.string(),
    tags: z.array(z.string()),
    created_at: z.date(),
    updated_at: z.date(),
    language: z.string(),
  }),
});

export const collections = { post: blogCollection };

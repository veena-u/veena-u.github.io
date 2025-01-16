import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
    site: 'https://veena-u.github.io',
    integrations: [tailwind(), react()],
    build: {
        assets: 'astro'
    },
    markdown: {
        shikiConfig: {
            theme: "material-theme-lighter",
            wrap: true,
        },
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex],
    },
});
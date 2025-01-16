/** @type {import('tailwindcss').Config} */
export default {
    content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
    darkMode: "class",
    theme: {
        extend: {
            colors: {
                pixel: {
                    background: "#f5f5f5",
                    primary: "#2b2e28",
                    light: "#bdbbb4", //D3D0c9
                    "code-bg": "#f5f5f5",
                    "menu-bg": "rgba(247, 247, 247, 0.9)",
                },
                // pixel: {
                //     background: "#2b2e28",
                //     primary: "#f7f7f7",
                //     light: "#4A4B4F",
                //     "code-bg": "#2b2e28",
                //     "menu-bg": "rgba(247, 247, 247, 0.9)",
                // }, //dark mode
                // pixel: {
                //     background: "#ddcfff",
                //     primary: "#040de1",
                //     light: "#937cfd",
                //     "code-bg": "#F8EDFF",
                //     "menu-bg": "rgba(248, 237, 255, 0.9)",
                // },
                "pixel-dark": {
                    background: "#ddcfff",
                    primary: "#040de1",
                    light: "#937cfd",
                    "code-bg": "#F8EDFF",
                    "menu-bg": "rgba(248, 237, 255, 0.9)",
                },
            },
            boxShadow: ({ theme }) => ({
                pixel: `6px 6px 0px -2px ${theme(
                    "colors.pixel.background"
                )}, 6px 6px ${theme("colors.pixel.primary")}`,
                "pixel-dark": `6px 6px 0px -2px ${theme(
                    "colors.pixel-dark.background"
                )}, 6px 6px ${theme("colors.pixel-dark.primary")}`,
            }),
            height: {
                404: "75vh",
            },
            textUnderlineOffset: {
                6: "6px",
            },
            backgroundImage: {
                "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
                "gradient-conic":
                    "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
            },
            ringOffsetWidth: {
                m2: "-2",
                m4: "-4",
            },
            borderWidth: {
                link: "2px",
            },
            gridTemplateColumns: {
                "1a1": "1fr auto 1fr",
                "1a": "1fr auto",
            },
            listStyleType: {
                none: 'none',
                disc: 'disc',
                decimal: 'decimal',
                circle: 'circle',
                square: 'square'
            },
        },
    },
    plugins: [],
};

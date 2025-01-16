import { markdownToTxt } from "markdown-to-txt";
import { type CollectionEntry } from "astro:content";

export function isoDate(date: Date): string {
    return date.toISOString().substring(0, 10);
}

export function md2txt(md: string) {
    return markdownToTxt(md);
}

export function postDateCompareReverse(
    a: CollectionEntry<"post">,
    b: CollectionEntry<"post">
) {
    return b.data.created_at.getTime() - a.data.created_at.getTime();
}

export function removePostSlugLang(slug: string) {
    return slug;
}
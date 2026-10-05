// DOM helpers shared by the pages. Text is always set with textContent, never as HTML.

import { ApiError } from "./api";

export type Child = Node | string | null | undefined | false;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Partial<Record<string, string>> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

export function showMessage(target: HTMLElement, text: string, kind: "error" | "ok" | "" = ""): void {
  target.textContent = text;
  target.className = `message ${kind}`;
}

export function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  return error instanceof Error ? error.message : String(error);
}

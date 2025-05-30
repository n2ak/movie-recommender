import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const roundRating = (rating: number) => {
  return Math.round(rating * 2) / 2;
};

export function timeSince(d: Date) {
  const seconds = Math.floor((Date.now() - d.getTime()) / 1000);
  function format(interval: number, s: string) {
    interval = Math.floor(interval);
    return `${interval} ${s}${interval === 1 ? "" : "s"}`;
  }
  let interval = seconds / 31536000;
  if (interval > 1) return format(interval, "year");
  interval = seconds / 2592000;
  if (interval > 1) return format(interval, "month");
  interval = seconds / 86400;
  if (interval > 1) return format(interval, "day");
  interval = seconds / 3600;
  if (interval > 1) return format(interval, "hour");
  interval = seconds / 60;
  if (interval > 1) return format(interval, "minute");
  return format(interval, "second");
}

export function joinCN(...args: string[]) {
  return args.join(" ");
}

export const formatNumber = (n: number) =>
  Intl.NumberFormat("en", { notation: "compact" }).format(n);

export const formatDate = (date: Date) =>
  date.toLocaleString(undefined, {
    day: "2-digit",
    year: "numeric",
    month: "long",
    minute: "2-digit",
    hour: "2-digit",
  });

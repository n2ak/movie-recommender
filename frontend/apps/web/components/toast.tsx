"use client";
import { CheckIcon, XIcon } from "lucide-react";
import { toast } from "sonner";

export function success(
  message: string,
  description?: string,
  duration?: number
) {
  doToast({ message, description, duration, icon: <CheckIcon /> });
}

export function error(
  message: string,
  description?: string,
  duration?: number
) {
  doToast({ message, description, duration, icon: <XIcon /> });
}

function doToast({
  message,
  description,
  duration,
  icon,
}: {
  message: string;
  description?: string;
  icon: React.ReactNode;
  duration?: number;
}) {
  toast(message, {
    description: description,
    icon,
    duration: duration || 2000,
  });
}

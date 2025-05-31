interface Props {
  placeholder?: string;
  value?: string;
  className?: string;
  onChange?: (v: string) => void;
  type: "password" | "email" | "text";
  error?: string;
  name?: string;
  label?: string;
  disabled?: boolean;
  addPasswordToggle?: boolean;
  hasError?: boolean;
}

export default function FormField({
  placeholder,
  className,
  type,
  value,
  onChange,
  error,
  name,
  label,
  disabled,
  addPasswordToggle,
  hasError,
}: Props) {
  const [show, setShow] = useState(false);
  if (addPasswordToggle && type === "password") {
    type = show ? "text" : "password";
  }
  if (hasError === undefined) {
    hasError = !!error;
  }
  return (
    <ColStack>
      {label && (
        <label
          htmlFor={name}
          className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
        >
          {label}
        </label>
      )}
      <div className="relative">
        <div className="">
          <input
            disabled={!!disabled}
            value={value}
            name={name}
            onChange={(e) => {
              if (onChange) onChange(e.target.value);
            }}
            className={joinCN(
              className || "",
              "focus:outline-hidden w-full flex-4/6 border border-black/30 focus:border-blue-500 rounded-md py-1.5 px-2 bg-gray-200 dark:bg-gray-700 dark:text-white",
              hasError ? "!border-red-600" : ""
            )}
            type={type}
            placeholder={placeholder}
          />
        </div>
        {addPasswordToggle && (
          <div
            className="absolute inset-y-0 end-0 flex items-center me-1 dark:hover:bg-gray-800 cursor-pointer my-1 px-1 rounded-sm dark:text-white"
            onClick={() => setShow((s) => !s)}
          >
            {show ? <EyeIcon /> : <EyeOffIcon />}
          </div>
        )}
      </div>
      {!!error && (
        <span className="text-red-600 font-medium text-sm">{error}</span>
      )}
    </ColStack>
  );
}
import { joinCN } from "@/lib/utils";
import { EyeIcon, EyeOffIcon } from "lucide-react";
import { useState } from "react";
import { ColStack } from "./Container";

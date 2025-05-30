import { joinCN } from "@/lib/utils";
import { Spinner } from "@radix-ui/themes";
import { PropsWithChildren } from "react";

export default function Button({
  type,
  loading,
  children,
  onClick,
  className,
}: {
  loading?: boolean;
  type?: "submit";
  className?: string;
  onClick?: () => void;
} & PropsWithChildren) {
  return (
    <button
      type={type}
      disabled={loading}
      className={joinCN(
        className || "",
        "disabled:cursor-not-allowed cursor-pointer items-center text-center font-medium rounded-lg text-sm w-full sm:w-auto px-5 py-2.5 bg-2 dark:bg-2 dark:hover:bg-1 hover:bg-1"
      )}
      onClick={(e) => {
        if (onClick) {
          e.preventDefault();
          onClick();
        }
      }}
    >
      <div className="text-center">
        {!loading ? <>{children}</> : <Spinner className="mx-auto" />}
      </div>
    </button>
  );
}

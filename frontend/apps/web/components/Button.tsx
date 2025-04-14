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
      //   disabled={loading}
      className={
        className +
        " flex items-center justify-center text-white cursor-pointer bg-blue-700 hover:bg-blue-800 font-medium rounded-lg text-sm w-full sm:w-auto px-5 py-2.5 text-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
      }
      onClick={onClick}
    >
      <div className="">
        {!loading ? <>{children}</> : <Spinner className="mx-auto" />}
      </div>
    </button>
  );
}

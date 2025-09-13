import type { PropsWithChildren, Ref } from "react";
type Props = {
  className?: string;
  ref?: Ref<HTMLDivElement>;
} & PropsWithChildren;

export function RowStack(props: Props) {
  return Custom(props, {
    customClass: "flex flex-row",
  });
}

export function ColStack(props: Props) {
  return Custom(props, {
    customClass: "flex flex-col",
  });
}

function Custom(
  { className, children, ref }: Props,
  {
    customClass,
  }: {
    customClass: string;
  }
) {
  className = `${className || ""} ${customClass}`;
  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}

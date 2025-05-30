import { PropsWithChildren, Ref } from "react";
type Props = {
  className?: string;
  ref?: Ref<HTMLDivElement>;
} & PropsWithChildren;
export function Container(props: Props) {
  return Custom(props, {
    customClass: "w-[90%] mx-auto",
  });
}

export function Divider(props: Props) {
  return Custom(props, {
    customClass: "my-[2px] min-h-[1px] bg-black/30",
  });
}

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

import { useEffect } from "react";

function isClickInside(ref: HTMLElement, { x, y }: { x: number; y: number }) {
  const rect = ref.getBoundingClientRect();
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}

export function useOnClickOutside(
  ref: React.RefObject<HTMLElement | null>,
  onClickOutside: () => void,
  except?: React.RefObject<HTMLElement | null>[]
) {
  if (!except) {
    except = [];
  }
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const { x, y } = event;
      if (ref.current) {
        if (!isClickInside(ref.current, { x, y })) {
          // if we click on avatar the menu closes and opens again
          if (
            !except?.some((e) =>
              e.current ? isClickInside(e.current, { x, y }) : false
            )
          )
            onClickOutside();
        }
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [ref, onClickOutside]);
}

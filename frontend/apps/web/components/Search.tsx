import { searchMovies } from "@/_lib/actions/action";
import { useOnClickOutside } from "@/hooks/useClickedOutside";
import { useDictionary } from "@/hooks/useLanguageStore";
import { useQuery } from "@tanstack/react-query";
import { debounce } from "lodash";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState } from "react";
import { ColStack } from "./Container";

export default function Search({ userId }: { userId: number }) {
  const dict = useDictionary();
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const { data: movies } = useQuery({
    initialData: [],
    queryKey: ["movie_search", { search }],
    queryFn: () => searchMovies(search, 5),
  });
  const debouncedSearch = useMemo(
    () =>
      debounce((q: string) => {
        setSearch(q);
      }, 500),
    []
  );
  const wrapperRef = useRef<HTMLDivElement>(null);
  useOnClickOutside(wrapperRef, () => {
    setMenuOpen(false);
  });
  // TODO combine open & input in same state
  return (
    <div className="relative w-5/12">
      <input
        type="text"
        value={input}
        onChange={(e) => {
          const q = e.target.value.trim().toLowerCase();
          setInput(q);
          setMenuOpen(true);
          debouncedSearch(q);
        }}
        placeholder={dict.searchAMovie}
        className="w-full h-full p-1 px-2 rounded-2xl bg-gray-100 dark:text-black"
        onFocus={() => setMenuOpen(true)}
      />
      {menuOpen && movies.length > 0 && (
        <ColStack
          ref={wrapperRef}
          className="absolute bg-white w-full rounded-b-2xl shadow-md pb-2.5"
        >
          {movies.map((m) => (
            <div key={m.id}>
              <SearchItem
                title={m.title}
                id={m.id}
                href={m.href}
                clear={() => {
                  setInput("");
                  setSearch("");
                  setMenuOpen(false);
                }}
              />
            </div>
          ))}
        </ColStack>
      )}
    </div>
  );
}
const SearchItem = ({
  title,
  id,
  href,
  clear,
}: {
  title: string;
  id: number;
  href: string;
  clear: () => void;
}) => {
  const router = useRouter();
  return (
    <div className="py-3 px-1 hover:bg-gray-100 hover:cursor-pointer dark:text-black">
      <div
        onClick={() => {
          clear();
          router.push(`/movie/${id}`);
        }}
        className="flex flex-row"
      >
        <img src={href} className="max-h-20 mr-2" />
        <div className="flex my-auto">{title}</div>
      </div>
    </div>
  );
};

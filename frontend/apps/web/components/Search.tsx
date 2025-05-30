import { useOnClickOutside } from "@/hooks/useClickedOutside";
import { useDictionary } from "@/hooks/useLanguageStore";
import useMovieSearch from "@/hooks/useMovieSearch";
import { debounce } from "lodash";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState } from "react";
import { ColStack } from "./Container";

export default function Search({ userId }: { userId: number }) {
  const dict = useDictionary();
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const movies = useMovieSearch(search);
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
  return (
    <div className="relative w-xs md:w-md md:max-w-lg">
      {/* <label
        htmlFor="default-search"
        className="mb-2 text-sm font-medium text-gray-900 sr-only dark:text-white"
      >
        Search
      </label> */}
      <div className="relative">
        <div className=" absolute inset-y-0 start-0 flex items-center ps-3 pointer-events-none">
          <svg
            className="w-4 h-4 text-gray-500 dark:text-gray-400"
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 20 20"
          >
            <path
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"
            />
          </svg>
        </div>
        <div className="">
          <input
            type="text"
            value={input}
            id="default-search"
            onChange={(e) => {
              const q = e.target.value.trim().toLowerCase();
              setInput(q);
              setMenuOpen(true);
              debouncedSearch(q);
            }}
            placeholder={dict.searchAMovie}
            className="focus:outline-hidden w-full py-2 ps-10 text-sm text-gray-900 border border-gray-300 rounded-md bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            onFocus={() => setMenuOpen(true)}
          />
        </div>
        <div className="z-0">
          {menuOpen && movies.length > 0 && (
            <ColStack
              ref={wrapperRef}
              className="absolute top-[99%] bg-white w-full rounded-b-md shadow-md"
            >
              {movies.map((m) => (
                <div key={m.id}>
                  <SearchItem
                    title={m.title}
                    id={m.id}
                    href={m.href}
                    genres={m.genres}
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
      </div>
    </div>
  );
}
const SearchItem = ({
  title,
  id,
  href,
  clear,
  genres,
}: {
  genres: string[];
  title: string;
  id: number;
  href: string;
  clear: () => void;
}) => {
  const router = useRouter();
  return (
    <div className="py-2 px-1 hover:bg-gray-100 rounded-b-md cursor-pointer dark:text-black group">
      <div
        onClick={() => {
          clear();
          router.push(`/movie/${id}`);
        }}
        className="flex gap-3 min-h-10 h-14 justify-start"
      >
        <img src={href} className="max-h-20 col-span-1 w-10 flex-none" />
        <div className="w-32 my-auto flex-10/12 overflow-ellipsis">
          <div className="flex flex-col gap-1 ">
            <h1 className="font-normal text-sm group-hover:underline">
              {title}
            </h1>
            <h3 className="font-normal text-xs text-slate-500">
              {genres.splice(0, 3).join(",")}
            </h3>
          </div>
        </div>
      </div>
    </div>
  );
};

import { ColStack } from "@/components/Container";
import { Recommended, RecommendedGenres } from "@/components/Recommendation";
import Skeleton from "@/components/Skeleton";
import { getMostWatchedGenres } from "@/lib/actions/movie";
import { Suspense } from "react";
import { TemperatureInput } from "./temp-input";


export default async function Home() {
  const genres = (await getMostWatchedGenres({})).data!.map(([g]) => g);

  return (
    <ColStack className="justify-center items-center">
      <div className="w-11/12 flex flex-col gap-2">
        <TemperatureInput />
        <Suspense fallback={<Skeleton nBoxes={5} />}>
          <Recommended />
        </Suspense>
        <Suspense fallback={<Skeleton nBoxes={5} />}>
          <RecommendedGenres genres={genres} />
        </Suspense>
      </div>
    </ColStack>
  );
}

// export const runtime = "nodejs";

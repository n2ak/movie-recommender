import { ColStack } from "@/components/Container";
import { Recommended, RecommendedGenres } from "@/components/Recommendation";
import Skeleton from "@/components/Skeleton";
import { Suspense } from "react";

export default async function Home() {
  return (
    <ColStack className="justify-center items-center">
      <div className="w-11/12 flex flex-col gap-2">
        <Suspense fallback={<Skeleton nBoxes={5} />}>
          <Recommended />
        </Suspense>
        <Suspense fallback={<Skeleton nBoxes={5} />}>
          <RecommendedGenres />
        </Suspense>
      </div>
    </ColStack>
  );
}

export const runtime = "nodejs";

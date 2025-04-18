import { auth } from "@/auth";
import { ColStack } from "@/components/Container";
import { Recommended, RecommendedGenres } from "@/components/Recommendation";
import Skeleton from "@/components/Skeleton";
import { Suspense } from "react";

export default async function Home() {
  // const user = useAuthStore(s=>s.user);
  const session = await auth();
  if (session) {
    const { user } = session;
    const userId = parseInt(user?.id as string);

    return (
      <ColStack className="justify-center items-center">
        <div className="w-11/12">
          <Suspense fallback={<Skeleton nbox={5} />}>
            <Recommended userId={userId} model="DLRM" />
          </Suspense>
          <Suspense fallback={<Skeleton nbox={5} />}>
            <RecommendedGenres userId={userId} />
          </Suspense>
        </div>
      </ColStack>
    );
  }
  return <>No session</>;
}

import { auth } from "@/auth";
import { ColStack } from "@/components/Container";
import { Box, Container, Skeleton, Stack } from "@mui/material";
import { Suspense } from "react";
import { Recommended, RecommendedGenres } from "./copms";

export default async function Home() {
  // const user = useAuthStore(s=>s.user);
  const session = await auth();
  if (session) {
    const { user } = session;
    const userId = parseInt(user?.id as string);

    return (
      <ColStack className="justify-center items-center">
        <div className="w-11/12">
          <Suspense fallback={<RowSkeleton nbox={5} />}>
            <Recommended userId={userId} model="DLRM" />
          </Suspense>
          <Suspense fallback={<RowSkeleton nbox={5} />}>
            <RecommendedGenres userId={userId} />
          </Suspense>
        </div>
      </ColStack>
    );
  }
  return <>No session</>;
}

function RowSkeleton({ nbox }: { nbox: number }) {
  return (
    <Container>
      <Stack direction={"row"}>
        {Array(nbox)
          .fill(0)
          .map((_, i) => (
            <Box sx={{ width: 210, marginRight: 0.5, my: 5 }} key={i}>
              <Skeleton variant="rectangular" sx={{ height: 100 }} />
            </Box>
          ))}
      </Stack>
    </Container>
  );
}

import { auth } from "@/auth";
import MovieCard, { MovieRow } from "@/components/MovieCard";
import { useSnackBar } from "@/components/SnackBarProvider";
import { getRecommendedMoviesForUser } from "@/lib/actions/action";
import {
  Container,
  Grid2 as Grid,
  Stack,
  ListItem as Item,
  Box,
  Skeleton,
} from "@mui/material";
import { Suspense } from "react";
import { Recommended, RecommendedGenre } from "./copms";
// import { useSession } from "next-auth/react";
// import { getServerSession } from "next-auth";

export default async function Home() {
  const session = await auth();
  if (session) {
    const { user } = session;
    const userId = parseInt(user?.id as string);

    return (
      <Container>
        <Stack spacing={2}>
          <Suspense fallback={<RowSkeleton nbox={5} />}>
            <Recommended userId={userId} model="MF" />
          </Suspense>
          <Suspense fallback={<RowSkeleton nbox={5} />}>
            <Recommended userId={userId} model="NCF" />
          </Suspense>
          <Suspense fallback={<RowSkeleton nbox={5} />}>
            <RecommendedGenre userId={userId} genres={["Action"]} model="NCF" />
          </Suspense>
          <Suspense fallback={<RowSkeleton nbox={5} />}>
            <RecommendedGenre userId={userId} genres={["Action"]} model="MF" />
          </Suspense>
          {/* <Suspense fallback={<RowSkeleton nbox={5} />}>
            <ContinueWatching userId={userId} />
          </Suspense> */}
        </Stack>
      </Container>
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

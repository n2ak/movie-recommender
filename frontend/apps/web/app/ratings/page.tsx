"use client";
import { RatingsTable } from "@/app/ratings/MovieRatingsList";
import { RowStack } from "@/components/Container";
import { useAuthStore } from "@/hooks/useAuthStore";
import { Container } from "@radix-ui/themes";
import { redirect, useSearchParams } from "next/navigation";

export default function RatingsPage() {
  const user = useAuthStore((s) => s.user);
  const searchParams = useSearchParams();
  const pageNumber = parseInt(searchParams.get("page") || "0");
  const count = parseInt(searchParams.get("count") || "10");
  if (isNaN(pageNumber) || isNaN(count)) {
    return redirect("/ratings");
  }
  if (!user) {
    return null;
  }
  return (
    <Container>
      <RowStack className="mb-5 w-full justify-between"></RowStack>
      <RatingsTable userId={user.id} pageNumber={pageNumber} count={count} />
    </Container>
  );
}

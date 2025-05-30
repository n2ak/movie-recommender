"use client";
import { RatingsTable } from "@/app/ratings/MovieRatingsList";
import { RowStack } from "@/components/Container";
import { useAuthStore } from "@/hooks/useAuthStore";
import { Container } from "@radix-ui/themes";

export default function RatingsPage() {
  const user = useAuthStore((s) => s.user);
  if (!user) {
    return null;
  }
  return (
    <Container>
      <RowStack className="mb-5 w-full justify-between"></RowStack>
      <RatingsTable userId={user.id} />
    </Container>
  );
}

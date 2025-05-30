import { DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import useMovieReview from "@/hooks/useMovieReview";
import {
  editMovieReviewAndRating,
  MovieWithUserRating,
} from "@/lib/actions/movie";
import { Dialog, Flex, Text, TextField } from "@radix-ui/themes";
import { useEffect, useState } from "react";
import { VarRating } from "./Rating";
import { error, success } from "./toast";
import { Button } from "./ui/button";
import { Checkbox } from "./ui/checkbox";

export default function EditMovieRatingAndReviewModal({
  movie,
  onClose,
  onSave,
}: {
  movie: MovieWithUserRating | undefined;
  onSave: () => void;
  onClose: () => void;
}) {
  const [state, setState] = useState({
    rating: 0,
    text: "",
    title: "",
  });
  const { review } = useMovieReview(movie?.id);
  useEffect(() => {
    if (review?.data)
      setState({
        rating: movie?.userRating[0]?.rating || 0,
        title: review.data.title,
        text: review.data.text,
      });
  }, [movie, review, movie?.userRating]);

  const [addReview, setAddReview] = useState(false);
  if (!movie) {
    return null;
  }

  const canToggleAdd = !review;

  return (
    <div className="z-10">
      <Dialog.Root open={true}>
        <Dialog.Content maxWidth="450px" className="flex flex-col gap-4">
          <Flex direction="column" gap="3">
            <Dialog.Title>Edit rating</Dialog.Title>
            <label>
              {/* <Text as="div" size="2" mb="1" weight="bold">
              Movie Name
            </Text> */}
              <TextField.Root value={movie.title} disabled />
            </label>
            <label className="flex gap-2">
              <Text as="div" size="2" mb="1" weight="bold">
                Your rating
              </Text>
              <VarRating
                v={state.rating}
                showValue
                onChange={(v) => {
                  setState({
                    ...state,
                    rating: v,
                  });
                }}
              />
            </label>
          </Flex>
          {canToggleAdd && (
            <div className="items-top flex space-x-2">
              <Checkbox
                id="addReview"
                checked={addReview}
                onCheckedChange={() => setAddReview(!addReview)}
              />
              <div className="leading-none">
                <label
                  htmlFor="addReview"
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  Accept terms and conditions
                </label>
              </div>
            </div>
          )}

          {(addReview || !canToggleAdd) && (
            <div>
              <DialogHeader>
                <DialogTitle>Review Movie</DialogTitle>
              </DialogHeader>
              <div className="py-4 flex flex-col gap-2">
                <Input
                  value={state.title}
                  placeholder="Title"
                  onChange={(v) =>
                    setState({
                      ...state,
                      title: v.target.value,
                    })
                  }
                />
                <Textarea
                  placeholder="type here"
                  // className="border-red-500"
                  value={state.text}
                  onChange={(v) =>
                    setState({
                      ...state,
                      text: v.target.value,
                    })
                  }
                />
              </div>
            </div>
          )}

          <Flex gap="3" justify="end">
            <Button onClick={() => onClose()} className="border">
              Cancel
            </Button>
            <Button
              className="border bg-black text-white"
              onClick={async () => {
                const res = await editMovieReviewAndRating({
                  movieId: movie.id,
                  ...state,
                  review_provided: addReview || !canToggleAdd,
                });
                if (!res.message) {
                  success("Movie rating edited");
                  onSave();
                } else {
                  error("Error: " + res.message);
                  console.error("Error rating movie", { res });
                }
              }}
            >
              Save
            </Button>
          </Flex>
        </Dialog.Content>
      </Dialog.Root>
    </div>
  );
}

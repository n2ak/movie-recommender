import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import useMovieReview from "@/hooks/useMovieReview";
import {
  editMovieReviewAndRating,
  MovieWithUserRating,
} from "@/lib/actions/movie";
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { VarRating } from "./Rating";
import { error, success } from "./toast";
import { Button } from "./ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "./ui/collapsible";
import { Label } from "./ui/label";
import { Separator } from "./ui/separator";
import { Textarea } from "./ui/textarea";

export default function EditMovieRatingAndReviewModal({
  movie,
  onClose,
  onSave,
}: {
  movie: MovieWithUserRating | undefined;
  onSave: () => void;
  onClose: () => void;
}) {
  const { review } = useMovieReview(movie?.id);
  const [rating, setRating] = useState(0);
  const [text, setText] = useState("");
  const [title, setTitle] = useState("");

  useEffect(() => {
    if (movie?.userRating[0]) setRating(movie.userRating[0].rating);
    if (review) {
      setText(review.text);
      setTitle(review.title);
    }
  }, [movie, review]);

  const [reviewToggle, setReviewToggle] = useState(false);
  if (!movie) {
    return null;
  }
  let canSave = false;
  if (rating !== 0) {
    // movie has been rated before
    const reviewIsValid = text !== "" && title !== "";
    const reviewIsEmpty = text === "" && title === "";
    const ratingHasChanged = movie.userRating[0]?.rating !== rating;
    if (review) {
      // movie has been reviewed before
      const reviewHasChanged = review.text !== text || review.title !== title;
      canSave = (ratingHasChanged || reviewHasChanged) && reviewIsValid;
    } else {
      // no review before
      if (reviewIsEmpty) {
        canSave = ratingHasChanged;
      } else {
        canSave = ratingHasChanged && reviewIsValid;
      }
    }
  }

  return (
    <div className="z-10">
      <Dialog open={true} onOpenChange={onClose}>
        <form>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Edit rating and review</DialogTitle>
              <DialogDescription>
                <Input type="text" value={movie?.title} disabled />
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-3">
              <Label htmlFor="name-1">Your rating</Label>
              <VarRating v={rating} showValue onChange={setRating} />
            </div>
            <Separator />
            <Collapsible
              open={reviewToggle}
              onOpenChange={setReviewToggle}
              className="flex w-[350px] flex-col gap-2"
            >
              <div className="flex items-center justify-between gap-4 px-4">
                <h4 className="text-sm font-semibold">Edit review</h4>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="icon" className="size-8">
                    {reviewToggle ? <ArrowUpIcon /> : <ArrowDownIcon />}
                    <span className="sr-only">Toggle</span>
                  </Button>
                </CollapsibleTrigger>
              </div>
              <CollapsibleContent className="flex flex-col gap-2">
                <div className="grid gap-4">
                  <div className="grid gap-3">
                    <Label htmlFor="name-1">Title</Label>
                    <Input
                      value={title}
                      placeholder="Title"
                      onChange={(v) => setTitle(v.target.value.trim())}
                    />
                  </div>
                  <div className="grid gap-3">
                    <Label htmlFor="username-1">Review body</Label>
                    <Textarea
                      placeholder="type here"
                      value={text}
                      onChange={(v) => setText(v.target.value.trim())}
                      className="max-h-[100px]"
                    />
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline" onClick={() => onClose()}>
                  Cancel
                </Button>
              </DialogClose>
              <Button
                onClick={async () => {
                  const reviewChanged = true;
                  console.log({ reviewChanged });
                  const res = await editMovieReviewAndRating({
                    text,
                    title,
                    rating,
                    reviewChanged,
                    movieId: movie?.id || 0,
                  });
                  if (!res.message) {
                    success("Movie rating edited");
                    onSave();
                  } else {
                    error("Error: " + res.message);
                    console.error("Error rating movie", { res });
                  }
                }}
                disabled={!canSave}
              >
                Save changes
              </Button>
            </DialogFooter>
          </DialogContent>
        </form>
      </Dialog>
    </div>
  );
}

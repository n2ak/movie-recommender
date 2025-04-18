import { rateMovie } from "@/_lib/actions/movie";
import { Dialog, Flex, Text, TextField } from "@radix-ui/themes";
import { useEffect, useState } from "react";
import type { MovieWithUserRating } from "../../../packages/database/src/movieDb";
import Button from "./Button";
import { useSnackBar } from "./providers/SnackBarProvider";
import { VarRating } from "./Rating";

export default function RateMovieModal({
  movie,
  onClose,
  onSave,
}: {
  movie: MovieWithUserRating | undefined;
  onSave: () => void;
  onClose: () => void;
}) {
  const [state, setState] = useState({
    open: false,
    rating: 0,
  });
  useEffect(() => {
    setState({
      open: !!movie,
      rating: movie?.userRating[0]?.rating || 0,
    });
  }, [!!movie]);
  const snackBar = useSnackBar();
  if (!movie) {
    return null;
  }

  return (
    <Dialog.Root open={state.open}>
      <Dialog.Content maxWidth="450px">
        <Dialog.Title>Edit rating</Dialog.Title>

        <Flex direction="column" gap="3">
          <label>
            <Text as="div" size="2" mb="1" weight="bold">
              Movie Name
            </Text>
            <TextField.Root value={movie.title} disabled />
          </label>
          <label>
            <Text as="div" size="2" mb="1" weight="bold">
              Your rating
            </Text>
            <VarRating
              v={state.rating}
              showValue
              onChange={(v) => {
                setState({
                  open: state.open,
                  rating: v,
                });
              }}
            />
          </label>
        </Flex>
        <Flex gap="3" mt="4" justify="end">
          <Button onClick={() => onClose()} className="!bg-white !text-black">
            Cancel
          </Button>
          <Button
            onClick={async () => {
              const res = await rateMovie(movie.id, state.rating);
              if (!res.errors && !res.message && res.message !== "") {
                snackBar.success("Movie rating edited", 2000);
                onSave();
              } else {
                snackBar.error("Error: " + res.message, 2000);
                console.error("Error rating movie", { res });
              }
            }}
          >
            Save
          </Button>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}

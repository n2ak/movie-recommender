import {
  Box,
  Button,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
} from "@mui/material";
import type { Movie as MovieModel } from "@repo/database";
import { useState } from "react";
import { FixedRating, VarRating } from "./Rating";

export default function MovieComp({ movie }: { movie: MovieModel }) {
  const [open, setOpen] = useState(false);
  const [personalRating, setPersonalRating] = useState(0);
  const rate = () => {
    // rateMovie(id, personalRating);
  };
  const avg_rating = Math.round(movie.avg_rating * 2) / 2;

  return (
    <>
      <Container>
        <Typography>{movie.title}</Typography>
        <Typography>
          Rating:
          <FixedRating v={avg_rating} />
          {avg_rating}
        </Typography>
        <Button variant="outlined" onClick={() => setOpen((o) => !o)}>
          Rate
        </Button>
        <Dialog
          // selectedValue={selectedValue}
          open={open}
          // onClose={handleClose}
        >
          <DialogTitle>Give '{movie.title}' your rating.</DialogTitle>
          <DialogContent>
            <Box
              noValidate
              component="form"
              sx={{
                display: "flex",
                flexDirection: "column",
                m: "auto",
                width: "fit-content",
              }}
            >
              <VarRating
                v={personalRating}
                onChange={(_, v) => setPersonalRating(v as number)}
              />
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => rate()}>Done</Button>
            <Button onClick={() => setOpen(false)}>Close</Button>
          </DialogActions>
        </Dialog>
      </Container>
    </>
  );
}

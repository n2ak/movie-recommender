import {
  Typography,
  Container,
  Rating,
  Button,
  Dialog,
  DialogTitle,
  Box,
  DialogContentText,
  DialogContent,
  DialogActions,
} from "@mui/material";
import { MovieModel } from "@prisma/client";
import { useState } from "react";

export default function Movie({ movie }: { movie: MovieModel }) {
  const [open, setOpen] = useState(false);
  const [personalRating, setPersonalRating] = useState(0);
  const rate = () => {
    console.log("Rating", movie.title, personalRating);
    // rateMovie(id, personalRating);
  };
  const avg_rating = Math.round(movie.avg_rating * 2) / 2;

  return (
    <>
      <Container>
        <Typography>{movie.title}</Typography>
        <Typography>
          Rating:
          <Rating readOnly value={avg_rating} precision={0.5}></Rating>
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
              <Rating
                value={personalRating}
                onChange={(_, v) => setPersonalRating(v as number)}
                precision={0.5}
              ></Rating>
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

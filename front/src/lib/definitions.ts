import type { MovieModel, UserModel, UserMovieRating } from "@prisma/client";

// export type User = Omit<UserModel, "password">;
export type User = UserModel;
export type Movie = MovieModel;
export type MovieRating = UserMovieRating;
export type MovieWithRating = Movie & {
  userRating: MovieRating[];
};
export type MovieForUser = MovieModel & {
  userRating: UserMovieRating | null;
};

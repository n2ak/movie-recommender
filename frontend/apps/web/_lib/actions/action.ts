"use server";
import { auth, signIn, signOut } from "@/auth";

import {
  Backend,
  type GetMoviesRecomResponse,
  type ModelType,
} from "@repo/backend_api";
import { Movie, moviesDb, type MovieGenre } from "@repo/database";
import authDb from "@repo/database/authDb";
import { AuthError } from "next-auth";
import { parseProfileSettings, parseRating } from "../validation";
import type { LoginFormState, ProfileSettingsFormState } from "./FormStates";

export async function changeProfileSettingsAction(
  data: ProfileSettingsFormState["data"]
): Promise<ProfileSettingsFormState> {
  const session = await auth();
  if (!session?.user || !session.user.id) {
    return {
      data: data,
      message: "Not authorized",
    };
  }
  const userId = parseInt(session.user.id);
  const { data: parsedData, errors } = parseProfileSettings(data);

  if (!errors) {
    try {
      const newData = await authDb.changeProfileSettings(userId, data);
      return {
        data: {
          name: newData.username,
          email: data.email,
        },
      };
    } catch {
      return {
        data,
        message: "Couldn't change settings",
      };
    }
  }
  console.log({ errors });
  return {
    errors: errors,
    data,
    message: errors.name,
  };
}
export async function authenticate(
  prevState: LoginFormState,
  formData: FormData
): Promise<LoginFormState> {
  try {
    await signIn("credentials", formData);
    return prevState;
  } catch (error) {
    if (error instanceof AuthError) {
      switch (error.type) {
        case "CredentialsSignin":
          return {
            data: prevState.data,
            message: "Invalid credentials.",
          };
        default:
          return {
            data: prevState.data,
            message: "Something went wrong.",
          };
      }
    }
    throw error;
  }
  // redirect("/");
  // return {};
}
export async function logOut() {
  await signOut({
    redirectTo: "/auth/login",
  });
}

export async function rateMovie(
  userId: number,
  movieId: number,
  rating: number
) {
  const { data, errors } = parseRating(rating);
  if (!!errors) {
    throw Error("Bad rating request");
  }
  return moviesDb.rateMovie(userId, movieId, rating);
}
export async function getRecommendedMoviesForUser(
  userId: number,
  model: ModelType
) {
  // TODO : get unwatched/unrated movie ids.
  const resp = await Backend.getMoviesRecom(userId, model, null);
  return parseResponse(resp);
}
async function parseResponse(response: GetMoviesRecomResponse) {
  const ret: {
    error:
      | undefined
      | {
          [index: string]: string;
        };
    result: {
      movies: Movie[] | undefined;
      predictions: number[] | undefined;
    };
  } = {
    error: undefined,
    result: {
      movies: undefined,
      predictions: undefined,
    },
  };
  if (response.status_code != 200) {
    ret.error = response.error;
  } else {
    ret.result.predictions = response.result.map((p) => p.predicted_rating);
    ret.result.movies = await moviesDb.getMovies(
      response.result.map((p) => p.movieId)
    );
  }
  return ret;
}
export async function getRecommendedGenreMovies(
  userId: number,
  genre: MovieGenre,
  model: ModelType
) {
  const resp = await Backend.getGenreRecom(userId, model, [genre], null);
  return parseResponse(resp);
}
export const getRatingsForUser = moviesDb.getRatingsForUser;
export const getNumberOfRatings = moviesDb.getNumberOfRatings;
export const getMovieForUser = moviesDb.getMovieForUser;
export const getMostWatchedGenres = async (userId: number) => {
  const genres = await getMostGenresRatings(userId);
  return genres.map((a) => a[0]).slice(0, 3);
};
export const getMostGenresRatings = async (userId: number) => {
  const genres = await moviesDb.getMostWatchedGenres(userId);
  return genres;
};
export const searchMovies = async (q: string, limit: number) => {
  const genres = await moviesDb.searchMovies(q, limit);
  return genres;
};

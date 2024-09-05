import "server-only";
import { prismaClient } from "./connect";
import { Movie, MovieForUser, MovieRating } from "../definitions";
import { Prisma } from "@prisma/client";
import { MovieGenre } from ".";

const moviesDb = {
  rateMovie: async function (userId: number, movieId: number, rating: number) {
    const a = await prismaClient.userMovieRating.findFirst({
      where: {
        movieModelId: movieId,
        userModelId: userId,
      },
    });
    if (!a) {
      console.log("movie Id", movieId);
      return await prismaClient.userMovieRating.create({
        data: {
          movieModelId: movieId,
          userModelId: userId,
          rating: rating,
          timestamp: new Date(),
        },
      });
    } else {
      return await prismaClient.userMovieRating.update({
        where: {
          id: a.id,
          movieModelId: movieId,
          userModelId: userId,
        },
        data: {
          rating: rating,
        },
      });
    }
  },
  getMovieRatingForUser: async function (
    userId: number,
    start: number,
    count: number
  ): Promise<Array<MovieRating>> {
    const ratings = await prismaClient.userMovieRating.findMany({
      where: {
        userModelId: userId,
      },
      orderBy: {
        rating: "desc",
      },
      skip: start,
      take: count,
      include: {
        movie: true,
      },
    });
    return ratings;
  },
  getGenresMovies: async function (genres: MovieGenre[]) {
    const movies = await prismaClient.movieModel.findMany({
      where: {
        genres: {
          hasSome: genres,
        },
      },
    });
    return movies;
  },
  getMovieForUser: async function (
    userId: number,
    movieId: number
  ): Promise<MovieForUser | null> {
    let result: MovieForUser | null = null;
    const movie = await prismaClient.movieModel.findFirst({
      where: {
        movieId: movieId,
      },
    });
    if (!!movie) {
      result = { ...movie, userRating: null };
      const rating = await prismaClient.userMovieRating.findFirst({
        where: {
          movieModelId: movieId,
          userModelId: userId,
        },
      });
      result.userRating = rating;
    }
    return result;
  },
  getMovies: async function (movieIds: number[]): Promise<Movie[]> {
    return await prismaClient.movieModel.findMany({
      where: {
        movieId: {
          in: movieIds,
        },
      },
    });
  },
  getRatingsForUser: async function (
    userId: number,
    start: number,
    count: number,
    sortby: RatingSortBy,
    order: Prisma.SortOrder
  ): Promise<RatingWithMovie[]> {
    let orderBy = undefined;
    switch (sortby) {
      case "timestamp":
      case "rating":
        orderBy = {
          [sortby]: order,
        };
        break;
      case "avg_rating":
        orderBy = [
          {
            movie: {
              avg_rating: order,
            },
          },
          {
            movie: {
              total_ratings: "desc",
            },
          },
        ];
        break;
      case "title":
        orderBy = {
          movie: {
            title: order,
          },
        };
        break;
      default:
        orderBy = undefined;
    }
    return await prismaClient.userMovieRating.findMany({
      where: {
        userModelId: userId,
      },
      skip: start,
      take: count,
      include: {
        movie: true,
      },
      orderBy: orderBy as any,
    });
  },
};
export type RatingWithMovie = MovieRating & {
  movie: Movie;
};
export type RatingSortBy = "rating" | "timestamp" | "title" | "avg_rating";
export default moviesDb;

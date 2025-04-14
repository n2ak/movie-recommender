import { Prisma } from "@prisma/client";
import { MovieGenre } from ".";
import { prismaClient } from "./connect";
import { Movie, MovieForUser, MovieRating } from "./definitions";
function counts(arr: string[]) {
  const count: { [index: string]: number } = {};

  for (const num of arr) {
    count[num] = count[num] ? count[num] + 1 : 1;
  }
  return count;
}
const moviesDb = {
  searchMovies: async function (i: string, limit: number) {
    if (i.trim() === "") return [];
    // TODO: add order by
    return await prismaClient.movieModel.findMany({
      where: {
        title: {
          contains: i,
          mode: "insensitive",
        },
      },
      take: limit,
    });
  },
  getMostWatchedGenres: async function (userId: number) {
    const genres = await prismaClient.userMovieRating.findMany({
      where: {
        userModelId: userId,
      },
      select: {
        movie: {
          select: {
            genres: true,
          },
        },
      },
    });
    const count = counts(genres.flatMap((m) => m.movie.genres));
    const sorted = Object.entries(count).sort((a, b) => b[1] - a[1]);
    return sorted;
  },
  rateMovie: async function (userId: number, movieId: number, rating: number) {
    const a = await prismaClient.userMovieRating.findFirst({
      where: {
        movieModelId: movieId,
        userModelId: userId,
      },
    });
    if (!a) {
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
        id: movieId,
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
        id: {
          in: movieIds,
        },
      },
    });
  },
  getAllRatingsForUser: async function (userId: number) {
    return await prismaClient.userMovieRating.findMany({
      where: {
        userModelId: userId,
      },
    });
  },
  getNumberOfRatings: async function (userId: number) {
    return (
      await prismaClient.userMovieRating.aggregate({
        where: {
          userModelId: userId,
        },
        _count: true,
      })
    )._count;
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
export { moviesDb };

import type { MovieModel, Prisma } from "@prisma/client";
import { prismaClient } from "./connect";
import type {
  MovieSortKey,
  UserRatingSortKey
} from "./selects/movie";
import {
  findMovie,
  findMoviesInIds,
  userRatingInclude
} from "./selects/movie";

function counts(arr: string[]) {
  const count: { [index: string]: number } = {};

  for (const num of arr) {
    count[num] = count[num] ? count[num] + 1 : 1;
  }
  return count;
}

export async function getMovieForUser(params: {
  userId: number;
  movieId: number;
}) {
  const movie = await prismaClient.movieModel.findFirst(
    findMovie(params.userId, params.movieId)
  );
  return movie;
}
export async function searchMovies(params: { i: string; limit: number }) {
  if (params.i.trim() === "") return [];
  return await prismaClient.movieModel.findMany({
    where: {
      title: {
        contains: params.i,
        mode: "insensitive",
      },
    },
    take: params.limit,
  });
}
export async function getMostWatchedGenres({ userId }: { userId: number }) {
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
}

export async function getMovies(movieIds: number[], userId: number) {
  return await prismaClient.movieModel.findMany(
    findMoviesInIds(movieIds, userId)
  );
}
export async function getMoviesGenres(movieIds: number[]) {
  return await prismaClient.movieModel.findMany({
    where: {
      tmdbId: { in: movieIds }
    },
    select: { genres: true }
  });
}


export async function getNumberOfRatings(params: { userId: number }) {
  return (
    await prismaClient.userMovieRating.aggregate({
      where: {
        userModelId: params.userId,
      },
      _count: true,
    })
  )._count;
}

export async function getRatedMoviesForUser(params: {
  userId: number;
  start: number;
  count: number;
  sortby: RatedMoviesRatingSortKey;
  order: Prisma.SortOrder;
}) {
  const orderBy = {
    ...(userRatingSsortKeys.includes(params.sortby as UserRatingSortKey)
      ? {
        [params.sortby]: params.order,
      }
      : {
        movie: {
          [params.sortby]: params.order,
        },
      }),
  };
  const res = await prismaClient.userMovieRating.findMany({
    where: {
      userModelId: params.userId,
    },
    select: {
      movie: {
        include: {
          ...userRatingInclude(params.userId),
          _count: {
            select: {
              reviews: true,
            },
          },
        },
      },
    },
    take: params.count,
    skip: params.start,
    orderBy,
  });
  return res.map((a) => a.movie);
}

export async function getUserBestMovies(params: { userId: number; count?: number }) {
  const res = await prismaClient.userMovieRating.findMany({
    where: {
      userModelId: params.userId,
    },
    orderBy: {
      rating: "desc",
    },
    take: params.count ?? 10,
    select: {
      movie: {
        select: {
          tmdbId: true
        }
      },
      rating: true,
    },
  });
  return res.map((a) => ({ tmdbId: a.movie.tmdbId, userRating: a.rating }));
}

export function getAllMovies(): Promise<MovieModel[]> {
  return prismaClient.movieModel.findMany();
}
const userRatingSsortKeys: UserRatingSortKey[] = ["rating"];
export type RatedMoviesRatingSortKey = UserRatingSortKey | MovieSortKey;

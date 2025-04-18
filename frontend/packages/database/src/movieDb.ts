import { Prisma } from "@prisma/client";
import { getById } from "./authDb";
import { prismaClient } from "./connect";
import {
  findMovie,
  MovieSortKey,
  userRatingInclude,
  UserRatingSortKey,
} from "./selects/movie";
import { userSelect } from "./selects/user";
function counts(arr: string[]) {
  const count: { [index: string]: number } = {};

  for (const num of arr) {
    count[num] = count[num] ? count[num] + 1 : 1;
  }
  return count;
}
// const moviesDb = {
export async function getMovieForUser(userId: number, movieId: number) {
  const movie = await prismaClient.movieModel.findFirst(
    findMovie(userId, movieId)
  );
  return movie;
}
export async function searchMovies(i: string, limit: number) {
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
}
export async function getMostWatchedGenres(userId: number) {
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
export async function rateMovie(
  userId: number,
  movieId: number,
  rating: number
) {
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
}
export async function getMovies(
  movieIds: number[],
  userId: number
): Promise<MovieWithUserRating[]> {
  return await prismaClient.movieModel.findMany({
    where: {
      id: {
        in: movieIds,
      },
    },
    include: userRatingInclude(userId),
  });
}
export async function getNumberOfRatings(userId: number) {
  return (
    await prismaClient.userMovieRating.aggregate({
      where: {
        userModelId: userId,
      },
      _count: true,
    })
  )._count;
}
export async function getNumberOfMovieReviews(movieId: number) {
  return (
    await prismaClient.movieReview.aggregate({
      where: {
        movieModelId: movieId,
      },
      _count: true,
    })
  )._count;
}
const userRatingSsortKeys: UserRatingSortKey[] = ["rating", "timestamp"];
const movieSortKeys: MovieSortKey[] = [];
export type RatedMoviesRatingSortKey = UserRatingSortKey | MovieSortKey;
export async function getRatedMoviesForUser(
  userId: number,
  start: number,
  count: number,
  sortby: RatedMoviesRatingSortKey,
  order: Prisma.SortOrder
): Promise<MovieWithUserRating[]> {
  // const res = await prismaClient.movieModel.findMany({
  //   ...findRatings(userId),
  //   take: count,
  //   skip: start,
  //   orderBy: {
  //     [sortby]: order,
  //   },
  // });
  // console.log({ res });
  // return res;
  // TODO performance?
  const orderBy = {
    ...(userRatingSsortKeys.includes(sortby as any)
      ? {
          [sortby]: order,
        }
      : {
          movie: {
            [sortby]: order,
          },
        }),
  };
  const res = await prismaClient.userMovieRating.findMany({
    where: {
      userModelId: userId,
    },
    select: {
      movie: {
        include: userRatingInclude(userId),
      },
    },
    take: count,
    skip: start,
    orderBy,
  });
  return res.map((a) => a.movie);
}
export const reviewMovie = async (
  userId: number,
  movieId: number,
  text: string
) => {
  return await prismaClient.movieReview.create({
    data: {
      text: text,
      movieModelId: movieId,
      userModelId: userId,
    },
  });
};
export const likeMovieReview = async (
  userId: number,
  movieReviewId: number
) => {
  return await prismaClient.reviewLike.create({
    data: {
      movieReviewId: movieReviewId,
      userModelId: userId,
    },
  });
};
export const getMovieReviews = async (
  currentUserId: number,
  movieId: number,
  start: number,
  count: number,
  sortKey: any,
  order: "asc" | "desc"
) => {
  const res = await prismaClient.movieReview.findMany({
    where: {
      movieModelId: movieId,
    },
    include: {
      user: {
        select: {
          ...userSelect,
          movieRatings: {
            where: {
              movieModelId: movieId,
            },
            take: 1,
          },
        },
      },
      _count: {
        select: {
          likes: true,
          comments: true,
        },
      },
      likes: {
        where: {
          userModelId: currentUserId,
        },
        select: {
          id: true,
        },
      },
    },
    skip: start,
    take: count,
  });
  const userId = 10;
  const userRating = (await getMovieForUser(1, 1108))!.userRating;
  res.push({
    createdAt: new Date(),
    id: 0,
    movieModelId: movieId,
    text: "xd ".repeat(100),
    updatedAt: new Date(),
    userModelId: userId,
    user: {
      ...(await getById(userId))!,
      movieRatings: userRating,
    },
    _count: {
      likes: 100000,
      comments: 13000,
    },
    likes: [
      // {
      //   id: currentUserId,
      // },
    ],
  });
  return res;
};
export const commentMovieReview = async (
  userId: number,
  movieReviewId: number,
  text: string
) => {
  return await prismaClient.reviewComment.create({
    data: {
      movieReviewId: movieReviewId,
      userModelId: userId,
      comment: text,
    },
  });
};
export const likeComment = async (userId: number, commentId: number) => {
  return await prismaClient.commentLike.create({
    data: {
      userModelId: userId,
      reviewCommentId: commentId,
    },
  });
};
// };
export type MovieWithUserRating = NonNullable<
  Awaited<ReturnType<typeof getMovieForUser>>
>;

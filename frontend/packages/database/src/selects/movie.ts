import { Prisma } from "@prisma/client";

export const findRatings = (userId: number) => {
  return Prisma.validator<Prisma.MovieModelFindManyArgs>()({
    where: {
      userRating: {
        some: {
          userModelId: userId,
        },
      },
    },
    include: userRatingInclude(userId),
  });
};
export const userRatingInclude = (userId: number) =>
  Prisma.validator<Prisma.MovieModelInclude>()({
    userRating: {
      where: {
        userModelId: userId,
      },
      take: 1,
    },
  });
export const findMovie = (userId: number, movieId: number) => {
  return Prisma.validator<Prisma.MovieModelFindFirstArgs>()({
    where: {
      id: movieId,
    },
    include: {
      ...userRatingInclude(userId),
      _count: {
        select: {
          reviews: true,
        },
      },
    },
  });
};
export const findMoviesInIds = (
  ids: number[],
  userId: number,
  take?: number,
  skip?: number
) => {
  return Prisma.validator<Prisma.MovieModelFindManyArgs>()({
    where: {
      id: { in: ids },
    },
    include: {
      ...userRatingInclude(userId),
      _count: {
        select: {
          reviews: true,
        },
      },
    },
    take,
    skip,
  });
};
export type MovieSortKey = keyof Prisma.MovieModelOrderByWithRelationInput;
export type UserRatingSortKey =
  keyof Prisma.UserMovieRatingOrderByWithRelationInput;

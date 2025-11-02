import type { Prisma } from "@prisma/client";
import { prismaClient } from "./connect";
import { userSelect } from "./selects/user";

function reviewInclude(movieId: number, userId: number) {
  return {
    user: {
      select: {
        ...userSelect,
        movieRatings: {
          where: {
            tmdbId: movieId,
          },
          take: 1,
        },
      },
    },
    reactions: {
      include: {},
      where: {
        userModelId: userId,
      },
      take: 1,
    },
  };
}
export const getMovieReview = async (params: {
  userId: number;
  movieId: number;
}) => {
  const review = await prismaClient.movieReview.findUnique({
    where: {
      tmdbId_userModelId: {
        userModelId: params.userId,
        tmdbId: params.movieId,
      },
    },
    include: reviewInclude(params.movieId, params.userId),
  });
  return review;
};
export const getMovieReviewById = async (params: {
  userId: number;
  reviewId: number;
  movieId: number;
}) => {
  const review = await prismaClient.movieReview.findUnique({
    where: {
      id: params.reviewId,
    },
    include: reviewInclude(params.movieId, params.userId),
  });
  return review;
};

export async function getNumberOfMovieReviews(params: { movieId: number }) {
  return (
    await prismaClient.movieReview.aggregate({
      where: {
        tmdbId: params.movieId,
      },
      _count: true,
    })
  )._count;
}

export const reviewMovie = async ({
  text,
  userId,
  movieId,
  title,
}: {
  userId: number;
  movieId: number;
  title: string;
  text: string;
}) => {
  text = text.trim();
  title = title.trim();
  return await prismaClient.movieReview.upsert({
    where: {
      tmdbId_userModelId: {
        userModelId: userId,
        tmdbId: movieId,
      },
    },
    create: {
      title,
      text: text,
      tmdbId: movieId,
      userModelId: userId,
      ndislikes: 0,
      nlikes: 0,
    },
    update: {
      text: text,
      title: title,
    },
  });
};

export const reactToMovieReview = async (params: {
  userId: number;
  movieReviewId: number;
  type: "LIKE" | "DISLIKE";
}) => {
  const reaction = await prismaClient.reviewReaction.findUnique({
    where: {
      movieReviewId_userModelId: {
        movieReviewId: params.movieReviewId,
        userModelId: params.userId,
      },
    },
  });
  const data: Prisma.MovieReviewUpdateInput = {};
  if (reaction) {
    if (reaction.type === params.type) {
      // should remove like/dislike
      if (params.type === "LIKE") data.nlikes = { decrement: 1 };
      else data.ndislikes = { decrement: 1 };
      await prismaClient.reviewReaction.delete({
        where: {
          id: reaction.id,
        },
      });
    } else {
      // update LIKE <==> DISLIKE
      await prismaClient.reviewReaction.update({
        where: {
          id: reaction.id,
        },
        data: {
          type: params.type,
        },
      });
      data.nlikes =
        params.type === "DISLIKE" ? { decrement: 1 } : { increment: 1 };
      data.ndislikes =
        params.type === "LIKE" ? { decrement: 1 } : { increment: 1 };
    }
  } else {
    // not reacted to before,create reaction
    await prismaClient.reviewReaction.create({
      data: {
        userModelId: params.userId,
        movieReviewId: params.movieReviewId,
        type: params.type,
      },
    });
    if (params.type === "LIKE") data.nlikes = { increment: 1 };
    else data.ndislikes = { increment: 1 };
  }
  await prismaClient.movieReview.update({
    where: {
      id: params.movieReviewId,
    },
    data,
  });
};
export const getMovieReviews = async (params: {
  userId: number;
  movieId: number;
  start: number;
  count: number;
  // sortKey: string; // TODO
  order: "asc" | "desc";
}) => {
  const res = await prismaClient.movieReview.findMany({
    where: {
      tmdbId: params.movieId,
    },
    include: reviewInclude(params.movieId, params.userId),
    skip: params.start,
    take: params.count,
  });
  return res;
};
export async function editMovieReviewAndRating({
  rating,
  text,
  title,
  movieId,
  reviewChanged,
  userId,
}: {
  movieId: number;
  reviewChanged: boolean;
  userId: number;
  rating: number;
  text: string;
  title: string;
}) {
  return await Promise.all([
    await prismaClient.userMovieRating.upsert({
      where: {
        tmdbId_userModelId: {
          tmdbId: movieId,
          userModelId: userId,
        },
      },
      create: {
        tmdbId: movieId,
        userModelId: userId,
        rating: rating
      },
      update: {
        rating: rating,
      },
    }),
    reviewChanged && reviewMovie({ userId, movieId, title, text }),
  ]);
}

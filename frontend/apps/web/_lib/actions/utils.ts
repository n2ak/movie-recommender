import { auth } from "@/auth";
import { PrismaClientInitializationError } from "@repo/database";
import { ValidationError } from "../validation";

export function actionWrapWithError<E = {}>() {
  return function <Fn extends (...args: any[]) => any>(action: Fn) {
    return async (
      ...args: Parameters<Fn>
    ): Promise<{
      data?: Awaited<ReturnType<Fn>>;
      errors?: E;
      message?: string;
    }> => {
      try {
        const out = await action(...args);
        return {
          data: out,
        };
      } catch (e) {
        console.error(e);
        if (e instanceof CustomError) {
          return {
            message: e.message,
          };
        } else if (e instanceof ValidationError) {
          return {
            message: e.message,
            errors: e.errors,
          };
        } else if (e instanceof PrismaClientInitializationError) {
          return {
            message: e.message,
          };
        }
        return {
          errors: {} as any,
        };
      }
    };
  };
}

// export function actionWrap<Fn extends (...args: any[]) => any, E = void>(
//   action: Fn
// ) {
//   return async (
//     ...args: Parameters<Fn>
//   ): Promise<{
//     data?: Awaited<ReturnType<Fn>>;
//     errors?: E;
//     message?: string;
//   }> => {
//     try {
//       const out = await action(...args);
//       return {
//         data: out,
//       };
//     } catch (e) {
//       if (e instanceof CustomError) {
//         return {
//           message: e.message,
//         };
//       } else if (e instanceof ValidationError) {
//         return {
//           message: e.message,
//           errors: e.errors,
//         };
//       }
//       return {};
//     }
//   };
// }

// type A<I, O, E> = (input: I) => Promise<any>;
export class CustomError extends Error {
  constructor(message: string) {
    // Need to pass `options` as the second parameter to install the "cause" property.
    super(message);
  }
}

export async function getUserId() {
  const session = await auth();
  if (!session || !session.user || !session.user.id) {
    throw new CustomError("Unauthenticated");
  }
  return {
    userId: parseInt(session.user.id),
  };
}

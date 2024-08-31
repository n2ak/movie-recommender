import type { NextAuthConfig } from "next-auth";

export const authConfig = {
  pages: {
    signIn: "/login",
  },
  callbacks: {
    authorized({ auth, request: { nextUrl, method } }) {
      // if (nextUrl.pathname.startsWith("/api/auth/session")) return true;
      const isLoggedIn = !!auth?.user;
      console.log("a message", isLoggedIn, nextUrl.pathname, method);
      const inLoginPage = nextUrl.pathname.startsWith("/auth/login");
      if (isLoggedIn && inLoginPage)
        return Response.redirect(new URL("/home", nextUrl));

      if (isLoggedIn) return true;
      if (inLoginPage) return true;
      return Response.redirect(new URL("/auth/login", nextUrl));
    },
    session({ session, token }) {
      session.user.email = token.email as string;
      session.user.id = token.id as string;
      return session;
    },
    jwt: async ({ token, user }) => {
      if (user) {
        token.email = user.email;
        token.id = user.id;
        // console.log("token", token);
      }
      return token;
    },
  },
  providers: [],
} satisfies NextAuthConfig;

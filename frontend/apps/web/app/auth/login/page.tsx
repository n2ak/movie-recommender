"use client";
import { useAuthStore } from "@/hooks/useAuthStore";
import Link from "next/link";
import * as React from "react";
// import { useFormState, useActionState } from "react-dom";
import { ColStack } from "@/components/Container";
import FormField from "@/components/FormField";
import { Button } from "@/components/ui/button";
import { useDictionary } from "@/hooks/useLanguageStore";
import { signIn } from "next-auth/react";

export default function SignIn() {
  const dict = useDictionary();
  const { loading } = useAuthStore();
  const [loginData, setLoginData] = React.useState<{
    username: string;
    password: string;
    message?: string;
  }>({
    username: "",
    password: "",
  });

  if (loading) {
    return null;
  }
  const login = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const res = await signIn("credentials", {
      usernameOrEmail: loginData.username,
      password: loginData.password,
      redirect: false,
    });
    if (res.error) {
      setLoginData({
        ...loginData,
        message: "Invalid credentials",
      });
    } else {
      window.location.href = "/home";
    }
  };
  return (
    <>
      <div className="flex justify-center text-black dark:text-white min-h-screen items-center bg-gray-100 dark:bg-gray-900">
        <div className="rounded-md bg-white dark:bg-gray-800 min-w-[400px] px-10 py-5 shadow-2xl mb-3">
          <h2 className="font-bold text-5xl w-full flex m-auto mb-10 text-gray-900 dark:text-white">
            <div className="mx-auto">{dict.signIn}</div>
          </h2>
          <form
            noValidate
            onSubmit={login}
            className="flex flex-col w-full gap-2"
          >
            <ColStack className="gap-4">
              <FormField
                type="email"
                name="email"
                placeholder="your@email.com"
                value={loginData.username}
                onChange={(v) => {
                  setLoginData({
                    ...loginData,
                    username: v,
                  });
                }}
                hasError={!!loginData.message}
                label="Email or username"
              />
              <FormField
                label="Password"
                type="password"
                name="password"
                value={loginData.password}
                onChange={(v) => {
                  setLoginData({
                    ...loginData,
                    password: v,
                  });
                }}
                hasError={!!loginData.message}
                addPasswordTogle
              />
              {loginData.message && (
                <div
                  className="flex h-8 items-end space-x-1 text-red-600 dark:text-red-400"
                  aria-live="polite"
                  aria-atomic="true"
                >
                  {loginData.message}
                </div>
              )}
              <Button className="text-white bg-blue-700 dark:bg-blue-600 hover:bg-blue-800 dark:hover:bg-blue-700">
                Sign In
              </Button>
              <div className="text-center text-gray-700 dark:text-gray-300">
                {"Don't have an account? "}
                <span>
                  <Link
                    href="/auth/register"
                    className="underline text-blue-900 dark:text-blue-400"
                  >
                    Sign up
                  </Link>
                </span>
              </div>
            </ColStack>
          </form>
        </div>
      </div>
    </>
  );
}

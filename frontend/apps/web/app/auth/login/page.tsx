"use client";
import { ColStack } from "@/components/Container";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { useAuthStore } from "@/hooks/useAuthStore";
import { zodResolver } from "@hookform/resolvers/zod";
import { Container } from "@radix-ui/themes";
import { signIn } from "next-auth/react";
import Link from "next/link";
import * as React from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

const FormSchema = z.object({
  usernameOrEmail: z.union([
    z.string().min(2, {
      message: "Username must be at least 6 characters.",
    }),
    z.string().email(),
  ]),
  password: z.string().min(6, {
    message: "Password must be at least 6 characters.",
  }),
});

export default function SignIn() {
  const { loading } = useAuthStore();
  const [error, setError] = React.useState<boolean>(false);

  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      usernameOrEmail: "hitchcockthelegend",
      password: "hitchcockthelegend",
    },
  });
  async function onSubmit(data: z.infer<typeof FormSchema>) {
    const res = await signIn("credentials", {
      ...data,
      redirect: false,
    });
    if (!res.error) {
      window.location.href = "/home";
    } else {
      setError(true);
    }
  }
  if (loading) {
    return null;
  }
  return (
    <Container className="mt-[100px]">
      <Card className="max-w-lg mx-auto">
        <CardHeader>
          <CardTitle className="text-xl w-full text-center">LOG IN</CardTitle>
        </CardHeader>
        <CardContent>
          <ColStack className="gap-4">
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="space-y-6"
              >
                <FormField
                  control={form.control}
                  name="usernameOrEmail"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Username</FormLabel>
                      <FormControl>
                        <Input placeholder="shadcn" {...field} />
                      </FormControl>
                      <FormDescription className="text-red-400">
                        {form.formState.errors.usernameOrEmail &&
                          form.formState.errors.usernameOrEmail?.message}
                      </FormDescription>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="password"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Password</FormLabel>
                      <FormControl>
                        <Input
                          placeholder="password"
                          {...field}
                          type="password"
                        />
                      </FormControl>
                      <FormDescription className="text-red-400">
                        {form.formState.errors.password &&
                          form.formState.errors.password?.message}
                      </FormDescription>
                    </FormItem>
                  )}
                />
                <FormDescription className="text-red-400">
                  {error && "Invalid credentials"}
                </FormDescription>
                <Button type="submit" className="cursor-pointer">
                  Submit
                </Button>
              </form>
            </Form>
            {error && (
              <div
                className="flex h-8 items-end space-x-1 text-red-600 dark:text-red-400"
                aria-live="polite"
                aria-atomic="true"
              >
                Invalid credentials
              </div>
            )}
          </ColStack>
        </CardContent>

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
      </Card>
    </Container>
  );
}

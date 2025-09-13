import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
} from "@/components/ui/form";
import type { UserInfo } from "@/hooks/useAuthStore";
import { useAuthStore } from "@/hooks/useAuthStore";
import { changeProfileSettingsAction } from "@/lib/actions/user";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { signOut, useSession } from "next-auth/react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import DeleteAccountModal from "../../components/DeleteAccountModal";
import { Button } from "../../components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../../components/ui/collapsible";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { Separator } from "../../components/ui/separator";

export default function SettingsSection({ user }: { user: UserInfo }) {
  return (
    <>
      <ProfileSettingsSection user={user} />
      <Separator />
      <DeleteAccountSection />
    </>
  );
}
const FormSchema = z.object({
  username: z
    .string({
      message: "Username should be characters",
    })
    .min(4, "Username is too short")
    .max(30, "username is too long"),
});
function ProfileSettingsSection({ user }: { user: UserInfo }) {
  const { update } = useSession();

  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      username: user.username,
    },
  });
  const { isPending: saving, mutateAsync: changeSettings } = useMutation({
    mutationFn: (data: z.infer<typeof FormSchema>) =>
      changeProfileSettingsAction(data),
  });
  async function onSubmit(data: z.infer<typeof FormSchema>) {
    const res = await changeSettings(data);
    if (res.data && !res.message) {
      window.location.reload();
      await update();
    }
  }
  return (
    <>
      <h2>Profile Settings</h2>
      <Form {...form}>
        <form
          className="flex flex-col gap-2 min-w-[400px]"
          onSubmit={form.handleSubmit(onSubmit)}
        >
          <div className="flex flex-col gap-2">
            <FormField
              control={form.control}
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Username</FormLabel>
                  <FormControl>
                    <Input placeholder="username" {...field} />
                  </FormControl>
                  <FormDescription className="text-red-400">
                    {form.formState.errors.username &&
                      form.formState.errors.username?.message}
                  </FormDescription>
                </FormItem>
              )}
            />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="email">Email</Label>
            <Input
              name="email"
              type="email"
              placeholder="Email"
              className="w-full !cursor-not-allowed"
              disabled
              value={user.email}
            />
          </div>
          <div>
            <Button
              className="!h-[40px] !w-[100px] float-right text-white disabled:bg-gray-400"
              disabled={saving || !form.formState.isDirty}
              type="submit"
            >
              {saving ? "Saving..." : "Save"}
            </Button>
          </div>
        </form>
      </Form>
    </>
  );
}
function DeleteAccountSection() {
  const [reviewToggle, setReviewToggle] = useState(false);

  return (
    <Collapsible
      open={reviewToggle}
      onOpenChange={setReviewToggle}
      className=""
    >
      <div className="flex w-full justify-end  items-center">
        <h4 className="text-sm font-semibold">Advanced</h4>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="icon" className="size-8">
            {reviewToggle ? <ArrowUpIcon /> : <ArrowDownIcon />}
          </Button>
        </CollapsibleTrigger>
      </div>
      <CollapsibleContent className="flex flex-col items-end">
        <DeleteAccountButton />
      </CollapsibleContent>
    </Collapsible>
  );
}
function DeleteAccountButton() {
  const [open, setOpen] = useState(false);
  const { clearUser } = useAuthStore();
  return (
    <div>
      <Button onClick={() => setOpen(true)} variant={"destructive"}>
        Delete account.
      </Button>
      <DeleteAccountModal
        open={open}
        onCancel={() => setOpen(false)}
        onDelete={async () => {
          clearUser();
          await signOut();
          setOpen(false);
          // await logOut();
        }}
      />
    </div>
  );
}

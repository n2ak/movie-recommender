import { useAuthStore, UserInfo } from "@/hooks/useAuthStore";
import { ProfileSettingsFormState } from "@/lib/actions/FormStates";
import { changeProfileSettingsAction } from "@/lib/actions/user";
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { signOut, useSession } from "next-auth/react";
import { useActionState, useState } from "react";
import DeleteAccountModal from "../../components/DeleteAccountModal";
import { error, success } from "../../components/toast";
import { Button } from "../../components/ui/button";
import { Card, CardHeader, CardTitle } from "../../components/ui/card";
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
      <Card className="gap-2 dark:bg-secondary-foreground p-4 rounded-lg shadow-sm">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold mb-4">
            Settings
          </CardTitle>
        </CardHeader>
        <Separator />
        <ProfileSettingsSection user={user} />
        <Separator />
        <DeleteAccountSection />
      </Card>
    </>
  );
}

function ProfileSettingsSection({ user }: { user: UserInfo }) {
  const { update } = useSession();

  const [saving, setSaving] = useState(false);
  const [state, formAction] = useActionState<
    ProfileSettingsFormState,
    FormData
  >(
    async (prevState, formData) => {
      const data = {
        username: (formData.get("username") as string) || "",
        email: prevState.data.email,
      };
      console.log("Saving...", data);

      const res = await changeProfileSettingsAction(data);
      setSaving(false);
      if (res.message) {
        error("Error: " + res.message);
      } else {
        success("Saved.");
        await update({
          name: data.username,
        });
      }
      return {
        data: {
          email: data.email,
          name: data.username,
        },
      };
    },
    {
      data: {
        name: user.username || "",
        email: user.email || "",
      },
    }
  );
  const [settings, setSettings] = useState(state.data);
  return (
    <>
      <h2>Profile Settings</h2>
      <form className="flex flex-col gap-2 min-w-[400px]" action={formAction}>
        <div className="flex flex-col gap-2">
          <Label htmlFor="username">Username</Label>
          <Input
            name="username"
            type="text"
            placeholder="Username"
            className="w-full"
            value={settings.name}
            onChange={(e) =>
              setSettings({
                ...settings,
                name: e.target.value,
              })
            }
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
            value={settings.email}
            onChange={(e) =>
              setSettings({
                ...settings,
                name: e.target.value,
              })
            }
          />
        </div>
        <div>
          <Button
            className="!h-[40px] !w-[100px] float-right text-white"
            disabled={saving}
            type="submit"
            // onClick={() => {
            //   setSaving(true)

            // }}
          >
            Save
          </Button>
        </div>
      </form>
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

import { deleteAccount } from "@/lib/actions/user";
import { Text } from "@radix-ui/themes";
import { useState } from "react";
import { ColStack, RowStack } from "./Container";
import Password from "./FormField";
import { Button } from "./ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Label } from "./ui/label";

export default function DeleteAccountModal({
  open,
  onCancel,
  onDelete,
}: {
  open: boolean;
  onCancel: () => void;
  onDelete?: () => void;
}) {
  const [proceed, setProceed] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const cancel = () => {
    onCancel();
    setProceed(false);
    setDeleting(false);
    setPassword("");
    setError("");
  };
  return (
    <Dialog open={open} onOpenChange={cancel}>
      <DialogContent>
        <DialogTitle className="!text-red-500">Delete your account</DialogTitle>

        <DialogHeader>
          <div className="font-medium text-sm">
            This acttion is <span className="!text-red-500">Destructive</span>{" "}
            and
            <span className="!text-red-500"> Irreversible!</span>
            <br />
            <br />
            You will not be able to login again using this account.
            <br />
            Make sure before proceeding.
          </div>
        </DialogHeader>
        {!proceed ? (
          <div className="flex justify-end">
            <Button variant="destructive" onClick={() => setProceed(true)}>
              Proceed!
            </Button>
          </div>
        ) : (
          <>
            <ColStack className="mt-2 gap-2">
              <Label htmlFor="passwordConfirm">Password</Label>
              <Password
                value={password}
                onChange={setPassword}
                addToggle
                name="passwordConfirm"
              />
              {error && <Text className="!text-destructive">{error}</Text>}
            </ColStack>
            <RowStack className=" gap-2 justify-end mt-5">
              <Button onClick={cancel} variant="outline">
                Cancel
              </Button>
              <Button
                disabled={deleting}
                onClick={() => {
                  if (password.trim() === "") {
                    setError("Password is required.");
                    return;
                  }
                  setDeleting(true);
                  (async function () {
                    const deletion = await deleteAccount({ password });
                    console.log({ deletion });
                    setDeleting(false);
                    if (deletion.data) {
                      if (onDelete) onDelete();
                      setProceed(false);
                    } else {
                      setError("Invalid password.");
                    }
                  })();
                }}
                variant="destructive"
              >
                Delete!
              </Button>
            </RowStack>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

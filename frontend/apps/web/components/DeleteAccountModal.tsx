import { deleteAccount } from "@/lib/actions/user";
import { Dialog } from "@radix-ui/themes";
import { useState } from "react";
import Button from "./Button";
import { ColStack, RowStack } from "./Container";
import FormField from "./FormField";

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
  return (
    <Dialog.Root open={open}>
      <Dialog.Content maxWidth="450px">
        <Dialog.Title className="!text-red-500">
          Delete your account
        </Dialog.Title>

        <div className="font-medium text-sm">
          This act is <span className="!text-red-500">Destructive</span> and
          <span className="!text-red-500"> Irreversible</span>!
          <br />
          <br />
          You will not be able to login again using this account.
          <br />
          Make sure before proceeding.
        </div>
        {!proceed ? (
          <div className="flex justify-end">
            <Button className="!bg-red-500" onClick={() => setProceed(true)}>
              Proceed!
            </Button>
          </div>
        ) : (
          <>
            <ColStack className="mt-2">
              <FormField
                type="password"
                placeholder="Password"
                label="Password"
                value={password}
                onChange={(v) => setPassword(v)}
                addPasswordTogle
              />
            </ColStack>
            <RowStack className=" gap-2 justify-end mt-5">
              <Button
                onClick={() => {
                  setProceed(false);
                  setDeleting(false);
                  onCancel();
                }}
                className="!bg-white !text-black"
              >
                Cancel
              </Button>
              <Button
                loading={deleting}
                onClick={() => {
                  setDeleting(true);
                  (async function () {
                    const done = await deleteAccount({ password });
                    setDeleting(false);
                    if (done) {
                      if (onDelete) onDelete();
                      setProceed(false);
                    } else {
                      // TODO
                    }
                  })();
                }}
                className="!bg-red-500"
              >
                Delete!
              </Button>
            </RowStack>
          </>
        )}
      </Dialog.Content>
    </Dialog.Root>
  );
}

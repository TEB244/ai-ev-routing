# Claude code setup instructions on Lab server

1. Ensure you have NodeJS and NPM installed

```
node -v
npm -v
```

(if not installed):

To install NPM:

``curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -``

To install NodeJS:

``sudo apt-get install -y nodejs``

2. Install Claude Code

Run:

``npm install -g @anthropic-ai/claude-code``

Check installed:

``claude --version``

3. Start Claude Code

Change into repo directory:

``cd ~/ai-ev-routing``

Initalize Claude Code:

``claude``

4. Configure Claude Code

This step requires you to configure claude code. You'll need an anthropic account, and you'll need to create an API key.
The terminal walks you through the steps needed to configure everything.

5. Use Claude Code

Now you can use Claude Code right from the terminal on the lab server!

6. Suspend Claude Code

If you want to suspend Claude Code, use CTRL+Z.

You can type `fg` to bring Claude Code back.

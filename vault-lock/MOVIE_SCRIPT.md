# How my finger lock's neural network works: stop-motion script

A stop-motion movie (about 5 minutes) with a voice-over. Each scene has
**what to film** and **what to say**. The voice-over is about 770 words; at a
relaxed pace that's about 5 minutes. Skipping the two scenes marked
*(optional)* makes it about 4½ minutes; for something closer to 3 minutes,
also shorten scenes 4, 7 and 9 to their first two sentences.

**Before you film the ending:** your real code would be on camera. Set a demo
code first (`python set_code.py 1 4 2`), film, then set your real one again.

---

## 1. The safe (0:00 – 0:15)

**Film:** the safe. A hand walks up (fingers "walking"), presses the button,
the UNO Q's LEDs go white, the hand shows three numbers, the servo swings
open. Real footage is fine here; it's the hook.

**Say:**
> This is my safe. There's no keypad. You press the button, show the camera
> three numbers with your fingers, and if they're right, the servo swings it
> open. The thing reading your fingers is a neural network I built from
> scratch. Here's how it works.

## 2. What the computer sees (0:15 – 0:35)

**Film:** a photo of your hand (print one of your `data/` pictures large).
Lay graph paper over it; colour the squares to match, then replace the
colours with numbers written in each square, a few squares at a time.

**Say:**
> The camera sees a picture, but a computer only sees numbers. I crop the
> middle of the frame and shrink it to 64 by 64 pixels. Each pixel has a red,
> a green and a blue value, so every picture is 12,288 numbers. The network's
> job is to turn those numbers into one answer: zero, one, two, three, four or
> five fingers.

## 3. One neuron (0:35 – 0:55)

**Film:** a paper circle (the neuron) with strings or arrows coming in from
three sticky notes (inputs). Each arrow gets a small number tag (a weight).
The numbers slide into the circle and a total appears. Then a cardboard ramp
shaped like ReLU: a negative total hits the flat part and becomes 0; a
positive one rides up the slope.

**Say:**
> The building block is a neuron. It multiplies each input by a weight, adds
> them all up, and passes the total through ReLU: negative becomes zero,
> positive goes straight through. That one bend is what lets a stack of these
> learn shapes, not just straight lines.

## 4. Convolution (0:55 – 1:30)

**Film:** graph paper with a simple drawing of a finger edge (dark squares
next to light ones). A 3×3 window (cut a square hole in a sticky note, or use
clear plastic with a 3×3 grid drawn on) slides one square per frame across
the drawing. On a second sheet, one number appears for each position; the
numbers are big where the window sits on the edge. At the end, the second
sheet "lights up" along the edge (colour those squares in).

**Say:**
> For pictures I use a convolution. Instead of one neuron looking at the
> whole image, a tiny 3 by 3 filter, just nine weights, slides across the
> picture one step at a time. At every spot it multiplies and adds, and writes
> one number onto a new map. If the filter has learned to find an edge, the
> map lights up wherever there's an edge, anywhere in the picture, using the
> same nine weights. That sharing is why my first layer needs only 448
> weights instead of almost 200,000.

## 5. Width *(optional)* (1:30 – 1:45)

**Film:** fan out a stack of 16 sheets (tracing paper is great), each with a
different coloured-in map: vertical edges, horizontal edges, dark spots,
skin-coloured areas. Then stacks of 32, 64 and 128 cards appear next to it.

**Say:**
> One filter finds one pattern, so each layer has many of them. That's the
> width. My first layer has 16 filters, so it makes 16 feature maps: edges at
> different angles, light against dark, skin against the background. Each
> layer after that doubles it: 32, then 64, then 128.

## 6. Max pooling (1:45 – 2:10)

**Film:** a 4×4 grid of numbers on paper. A 2×2 paper frame lands on the
top-left block; the biggest number lifts off (a sticky note) and moves to a
2×2 grid beside it. Repeat for the other three blocks. Then nudge a "finger"
(a strip of paper) one square sideways and show the pooled grid barely
changes.

**Say:**
> After each convolution comes max pooling. I look at every 2 by 2 block and
> keep only the biggest number. The map shrinks to half its width and height,
> 64, then 32, 16 and 8, so there's far less to compute. And because it keeps
> the strongest signal, a finger that moves over by one pixel gives almost the
> same answer.

## 7. Depth (2:10 – 2:35)

**Film:** a tower of four blocks, labelled conv 1 to conv 4, built one at a
time. Next to it, on the printed hand photo, a paper circle grows with each
block: 3 pixels, 8, 18, then 38, which covers the whole hand.

**Say:**
> Stacking layers is the depth. My network has four convolution layers, and
> each one looks at the one below it, so it sees a bigger patch of the
> original picture: 3 pixels, then 8, then 18, then 38, enough to see a whole
> hand. Edges become bits of finger, bits become fingers, and fingers become
> "three in a row". In my tests, depth mattered most: with only two layers,
> the network couldn't see a whole hand and got less than half right.

## 8. The answer (2:35 – 2:50)

**Film:** the 128 cards from scene 5 each get squashed into one bead or LEGO
stud (global average pooling). The beads pour into six cups labelled 0 to 5;
one cup fills up much more than the others (that's the answer). End on a card
reading "98,694 weights".

**Say:**
> At the end, global average pooling squashes each of the 128 maps down to
> one number, and a last layer turns those into six scores, one for each
> count. Softmax turns the scores into probabilities, and the biggest one is
> the answer. Altogether, the network has 98,694 weights.

## 9. Learning from mistakes: backpropagation (2:50 – 3:20)

**Film:** a picture of three fingers goes in; the cups show "4" with a red ✗.
Then small arrows march backwards from the cups through the block tower
(right to left), and little paper dials (weights) on each block turn a
notch, some up, some down.

**Say:**
> At first the weights are random, so the answers are garbage. Training fixes
> that. I show it a picture and the right answer, and the loss measures how
> wrong it was. Backpropagation then works backwards through every layer
> using the chain rule, and figures out, for every single weight: if I nudge
> you a little, does the loss go up or down, and by how much?

## 10. Gradient descent (3:20 – 3:40)

**Film:** a hill made of paper or clay with a marble on top. The marble moves
down a little each frame and zig-zags into the valley. Label the hill
"loss" and the valley "right answers".

**Say:**
> Gradient descent uses that answer to move every weight a small step
> downhill. Do that thousands of times, on batches of 64 pictures, and the
> loss rolls down into a valley where the answers are right. I wrote that
> step myself, with momentum, so it keeps rolling through the small bumps
> instead of getting stuck.

## 11. The data *(optional)* (3:40 – 4:00)

**Film:** a pile of tiny printed pictures of your hand. Some flip over, turn
sideways, get bigger, get darker (augmentation). A few go into a separate box
labelled "test: never seen".

**Say:**
> It learned from about 2,500 pictures of my hand, taken with the webcam.
> During training every picture gets randomly turned, zoomed, mirrored and
> re-lit, so it can't just memorise them. Then I test it on recordings it has
> never seen: it gets 95 percent right, and every count from one to five was
> perfect.

## 12. Something new: Grad-CAM (4:00 – 4:30)

**Film:** print `results/gradcam_watch.png`. Its two left pictures are from
your first training: a fist wearing the watch, labelled "is 0, says 1", with
the heat right on the watch screen. Print `results/gradcam.png` too (heat on
the fingers). A paper "spotlight" lands on the watch, then a red ✗; then the
second print slides in and the spotlight moves onto the fingers.

**Say:**
> Here's something we didn't do in class: Grad-CAM. It uses backpropagation
> again, but asks a different question: which pixels made you say that? The
> answer is a heat map. And it caught my network cheating. It had decided my
> smartwatch meant "one", because I wore it in my "one" pictures. I recorded
> more pictures with and without the watch, and now the heat sits on the
> fingers, where it belongs.

## 13. On the Arduino (4:30 – 4:50)

**Film:** a paper copy of the block tower "walks" from the laptop into the
UNO Q. Then the real safe: button, white LEDs, three numbers (your demo
code), green LEDs, the servo opens.

**Say:**
> Last, I moved the trained network onto the Arduino UNO Q. It reads the
> camera about fifteen times a second, and the network takes about 25
> milliseconds per picture. So: press the button, the lights come on, one…
> four… two… and it's open.

(Change "one… four… two" to whatever demo code you film with.)

---

## Props

- Graph paper, sticky notes, a 3×3 window (sticky note with a hole, or clear
  plastic), tracing paper
- A printed photo of your hand (any picture from `data/`, printed large)
- Blocks or LEGO for the layer tower, beads or studs, six small cups
- A paper or clay hill and a marble
- Prints of `results/gradcam_watch.png`, `results/gradcam.png`, and a few
  pictures from `data/`
- The safe itself for scenes 1 and 13

## Filming tips

- Stop motion at 12 frames per second: move things a little between shots,
  and hold each important frame for 6–12 frames (½–1 s) so viewers can read
  it.
- Fix the camera (a tripod or a phone stand) and the lighting, so only the
  props change between shots.
- Record the voice-over first and time the scenes to it; the times above
  are a guide.
- The real numbers are in [CONCEPTS.md](CONCEPTS.md) if your teacher asks
  where they come from.

package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/gdamore/tcell/v2"
)

const (
	winningScore   = 7
	fps            = 60
	paddleH        = 5
	ballSpeed0     = 0.55
	speedInc       = 0.04
	maxBallSpeed   = 1.4
	aiSpeed        = 0.85
	aiReaction     = 0.80
	paddleMaxVel   = 1.6
	paddleAccel    = 0.4
	paddleFriction = 0.70
	keyHoldFrames  = 4

	physSteps = 3
	trailLen  = 6
	numSparks = 4
)

var frameDur = time.Second / fps

var (
	styleBall   = tcell.StyleDefault.Foreground(tcell.ColorWhite).Bold(true)
	stylePaddle = tcell.StyleDefault.Foreground(tcell.ColorTeal).Bold(true)
	styleScore  = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	styleNet    = tcell.StyleDefault.Foreground(tcell.ColorNavy)
	styleText   = tcell.StyleDefault.Foreground(tcell.ColorGreen).Bold(true)
	styleTrail  = tcell.StyleDefault.Foreground(tcell.ColorSilver)
	styleWin    = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	styleSpark  = tcell.StyleDefault.Foreground(tcell.ColorRed)
	styleBorder = tcell.StyleDefault.Foreground(tcell.ColorNavy)
	styleHint   = tcell.StyleDefault.Foreground(tcell.ColorNavy)
)

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func drawStr(s tcell.Screen, x, y int, text string, style tcell.Style) {
	for i, ch := range []rune(text) {
		s.SetContent(x+i, y, ch, nil, style)
	}
}

func drawCentered(s tcell.Screen, w, y int, text string, style tcell.Style) {
	runes := []rune(text)
	x := (w - len(runes)) / 2
	if x < 0 {
		x = 0
	}
	for i, ch := range runes {
		s.SetContent(x+i, y, ch, nil, style)
	}
}

func drawBorder(s tcell.Screen, w, h int) {
	for x := 0; x < w; x++ {
		s.SetContent(x, 0, tcell.RuneHLine, nil, styleBorder)
		s.SetContent(x, h-1, tcell.RuneHLine, nil, styleBorder)
	}
	for y := 0; y < h; y++ {
		s.SetContent(0, y, tcell.RuneVLine, nil, styleBorder)
		s.SetContent(w-1, y, tcell.RuneVLine, nil, styleBorder)
	}
	s.SetContent(0, 0, tcell.RuneULCorner, nil, styleBorder)
	s.SetContent(w-1, 0, tcell.RuneURCorner, nil, styleBorder)
	s.SetContent(0, h-1, tcell.RuneLLCorner, nil, styleBorder)
	s.SetContent(w-1, h-1, tcell.RuneLRCorner, nil, styleBorder)
}

func drawNet(s tcell.Screen, w, h int) {
	cx := w / 2
	for r := 1; r < h-1; r++ {
		ch := rune(' ')
		if r%2 == 0 {
			ch = '┊'
		}
		s.SetContent(cx, r, ch, nil, styleNet)
	}
}

func drawHUD(s tcell.Screen, p1Score, p2Score, w int, p1label, p2label string) {
	hud := fmt.Sprintf(" %s  %2d  ─ vs ─  %2d  %s ", p1label, p1Score, p2Score, p2label)
	drawCentered(s, w, 0, hud, styleScore)
}

func centeredLines(s tcell.Screen, w, h int, lines []string, style tcell.Style) {
	sy := h/2 - len(lines)/2
	for i, line := range lines {
		drawCentered(s, w, sy+i, line, style)
	}
}

type Spark struct{ r, c, ttl int }

type Point struct{ r, c int }

type Paddle struct {
	x, y   float64
	vy     float64
	height int
	score  int
}

func newPaddle(x, cy int) *Paddle {
	return &Paddle{
		x:      float64(x),
		y:      float64(cy - paddleH/2),
		height: paddleH,
	}
}

func (p *Paddle) top() int    { return int(p.y) }
func (p *Paddle) bottom() int { return int(p.y) + p.height - 1 }

func (p *Paddle) accelerate(dir float64, maxY int) {
	p.vy += dir * paddleAccel
	p.vy = clamp(p.vy, -paddleMaxVel, paddleMaxVel)
	p.applyVel(maxY)
}

func (p *Paddle) coast(maxY int) {
	p.vy *= paddleFriction
	if math.Abs(p.vy) < 0.05 {
		p.vy = 0
	}
	p.applyVel(maxY)
}

func (p *Paddle) aiStep(dir float64, maxY int) {
	p.vy = dir * aiSpeed
	p.applyVel(maxY)
}

func (p *Paddle) applyVel(maxY int) {
	p.y += p.vy
	p.y = clamp(p.y, 1, float64(maxY-p.height-1))
}

func (p *Paddle) draw(s tcell.Screen) {
	for i := 0; i < p.height; i++ {
		ch := rune('█')
		if i == 0 || i == p.height-1 {
			ch = '▓'
		}
		s.SetContent(int(p.x), int(p.y)+i, ch, nil, stylePaddle)
	}
}

type Ball struct {
	x, y   float64
	dx, dy float64
	speed  float64
	trail  []Point
	sparks []Spark
}

func newBall(cx, cy int) *Ball {
	b := &Ball{}
	b.reset(cx, cy)
	return b
}

func (b *Ball) reset(cx, cy int) {
	b.x = float64(cx)
	b.y = float64(cy)
	angle := (rand.Float64()*0.5 - 0.25) * math.Pi
	dirs := []float64{-1, 1}
	dir := dirs[rand.Intn(2)]
	b.dx = ballSpeed0 * math.Cos(angle) * dir
	b.dy = ballSpeed0 * math.Sin(angle)
	b.speed = ballSpeed0
	b.trail = b.trail[:0]
	b.sparks = b.sparks[:0]
}

func (b *Ball) addSpark(r, c int) {
	for i := 0; i < numSparks; i++ {
		b.sparks = append(b.sparks, Spark{
			r:   r + rand.Intn(3) - 1,
			c:   c + rand.Intn(3) - 1,
			ttl: 3,
		})
	}
}

func (b *Ball) updateTrail() {
	b.trail = append(b.trail, Point{int(b.y), int(b.x)})
	if len(b.trail) > trailLen {
		b.trail = b.trail[1:]
	}
	alive := b.sparks[:0]
	for _, sp := range b.sparks {
		sp.ttl--
		if sp.ttl > 0 {
			alive = append(alive, sp)
		}
	}
	b.sparks = alive
}

func (b *Ball) accelerate() {
	b.speed = math.Min(b.speed+speedInc, maxBallSpeed)
	cur := math.Hypot(b.dx, b.dy)
	if cur > 0 {
		r := b.speed / cur
		b.dx *= r
		b.dy *= r
	}
}

func (b *Ball) draw(s tcell.Screen) {
	for _, sp := range b.sparks {
		s.SetContent(sp.c, sp.r, '·', nil, styleSpark)
	}
	n := len(b.trail)
	for i, pt := range b.trail {
		fade := float64(i) / float64(n)
		ch := rune('·')
		if fade > 0.6 {
			ch = '●'
		} else if fade > 0.3 {
			ch = '•'
		}
		s.SetContent(pt.c, pt.r, ch, nil, styleTrail)
	}
	s.SetContent(int(b.x), int(b.y), '●', nil, styleBall)
}

func doAI(p *Paddle, b *Ball, maxY int) {
	if rand.Float64() > aiReaction {
		return
	}
	mid := p.y + float64(p.height)/2
	if b.y < mid-0.5 {
		p.aiStep(-1, maxY)
	} else if b.y > mid+0.5 {
		p.aiStep(1, maxY)
	}
}

func scoreFlash(s tcell.Screen, w, h int, scorer string) {
	msg := fmt.Sprintf("  ✦  %s scores!  ✦  ", scorer)
	for i := 0; i < 14; i++ {
		s.Clear()
		drawBorder(s, w, h)
		drawCentered(s, w, h/2, msg, styleWin)
		s.Show()
		time.Sleep(70 * time.Millisecond)
		s.Clear()
		drawBorder(s, w, h)
		s.Show()
		time.Sleep(40 * time.Millisecond)
	}
}

func winnerScreen(s tcell.Screen, evCh <-chan tcell.Event,
	p1Score, p2Score, w, h int, winner, betOn string) bool {

	lines := []string{
		fmt.Sprintf("  \U0001f3c6  %s WINS!  \U0001f3c6  ", winner),
		fmt.Sprintf("  %d  —  %d  ", p1Score, p2Score),
	}
	if betOn != "" {
		if betOn == winner {
			lines = append(lines, "  ✓  You won your bet!  ✓  ")
		} else {
			lines = append(lines, "  ✗  You lost your bet.  ✗  ")
		}
	}
	lines = append(lines, "", "  R: replay   Q: quit  ")

	s.Clear()
	drawBorder(s, w, h)
	centeredLines(s, w, h, lines, styleWin)
	s.Show()

	for ev := range evCh {
		ke, ok := ev.(*tcell.EventKey)
		if !ok {
			continue
		}
		switch {
		case ke.Rune() == 'r' || ke.Rune() == 'R':
			return true
		case ke.Rune() == 'q' || ke.Rune() == 'Q' || ke.Key() == tcell.KeyEscape:
			return false
		}
	}
	return false
}

type InputState struct {
	p1Up, p1Down         bool
	p2Up, p2Down         bool
	quit, pause, restart bool
}

type stickyKey struct {
	dir   float64
	timer int
}

func (sk *stickyKey) press(dir float64) {
	sk.dir = dir
	sk.timer = keyHoldFrames
}

func (sk *stickyKey) tick() float64 {
	if sk.timer > 0 {
		sk.timer--
		return sk.dir
	}
	sk.dir = 0
	return 0
}

func runGame(s tcell.Screen, mode int, betOn string) {
	w, h := s.Size()
	cx, cy := w/2, h/2

	var p1label, p2label string
	switch mode {
	case 2:
		p1label, p2label = "P1", "P2"
	case 3:
		p1label, p2label = "Airi", "Bo"
	default:
		p1label, p2label = "P1", "Airi"
	}

	p1 := newPaddle(2, cy)
	p2 := newPaddle(w-3, cy)
	ball := newBall(cx, cy)

	serveWait := 1.2
	serveTimer := serveWait
	paused := false

	var sk1, sk2 stickyKey

	evCh := make(chan tcell.Event, 64)
	go func() {
		for {
			ev := s.PollEvent()
			if ev == nil {
				close(evCh)
				return
			}
			evCh <- ev
		}
	}()

	ticker := time.NewTicker(frameDur)
	defer ticker.Stop()

	for range ticker.C {
		w, h = s.Size()
		cx, cy = w/2, h/2

		inp := InputState{}
	drainLoop:
		for {
			select {
			case ev, ok := <-evCh:
				if !ok {
					return
				}
				switch e := ev.(type) {
				case *tcell.EventResize:
					s.Sync()
					w, h = s.Size()
				case *tcell.EventKey:
					switch {
					case e.Key() == tcell.KeyEscape || e.Rune() == 'q' || e.Rune() == 'Q':
						inp.quit = true
					case e.Rune() == 'p' || e.Rune() == 'P':
						inp.pause = true
					case e.Rune() == 'r' || e.Rune() == 'R':
						inp.restart = true
					case e.Rune() == 'w' || e.Rune() == 'W':
						inp.p1Up = true
						sk1.press(-1)
					case e.Rune() == 's' || e.Rune() == 'S':
						inp.p1Down = true
						sk1.press(1)
					case e.Key() == tcell.KeyUp:
						inp.p2Up = true
						sk2.press(-1)
					case e.Key() == tcell.KeyDown:
						inp.p2Down = true
						sk2.press(1)
					}
				}
			default:
				break drainLoop
			}
		}

		if inp.quit {
			return
		}
		if inp.pause {
			paused = !paused
		}
		if inp.restart {
			p1.score, p2.score = 0, 0
			p1.vy, p2.vy = 0, 0
			ball.reset(cx, cy)
			serveTimer = serveWait
		}

		if paused {
			s.Clear()
			drawBorder(s, w, h)
			centeredLines(s, w, h, []string{"── PAUSED ──", "P to resume"}, styleText)
			s.Show()
			continue
		}

		if mode == 3 {
			doAI(p1, ball, h)
			doAI(p2, ball, h)
		} else {
			d1 := sk1.tick()
			if d1 != 0 {
				p1.accelerate(d1, h)
			} else {
				p1.coast(h)
			}

			if mode == 2 {
				d2 := sk2.tick()
				if d2 != 0 {
					p2.accelerate(d2, h)
				} else {
					p2.coast(h)
				}
			} else {
				doAI(p2, ball, h)
			}
		}

		if serveTimer > 0 {
			serveTimer -= frameDur.Seconds()
		} else {
			ball.updateTrail()

			scored := false
			for step := 0; step < physSteps && !scored; step++ {
				ball.x += ball.dx / physSteps
				ball.y += ball.dy / physSteps

				bx := int(ball.x)
				by := int(ball.y)

				if ball.y <= 1 {
					ball.y = 1
					ball.dy = math.Abs(ball.dy)
				}
				if ball.y >= float64(h-2) {
					ball.y = float64(h - 2)
					ball.dy = -math.Abs(ball.dy)
				}

				if ball.dx < 0 && bx == int(p1.x)+1 && by >= p1.top() && by <= p1.bottom() {
					ball.addSpark(by, bx)
					rel := (ball.y - (p1.y + float64(p1.height)/2)) / (float64(p1.height) / 2)
					angle := rel * math.Pi * 0.38
					spd := math.Hypot(ball.dx, ball.dy)
					ball.dx = math.Abs(spd * math.Cos(angle))
					ball.dy = spd * math.Sin(angle)
					ball.x = p1.x + 2
					ball.accelerate()
				}

				if ball.dx > 0 && bx == int(p2.x)-1 && by >= p2.top() && by <= p2.bottom() {
					ball.addSpark(by, bx)
					rel := (ball.y - (p2.y + float64(p2.height)/2)) / (float64(p2.height) / 2)
					angle := rel * math.Pi * 0.38
					spd := math.Hypot(ball.dx, ball.dy)
					ball.dx = -math.Abs(spd * math.Cos(angle))
					ball.dy = spd * math.Sin(angle)
					ball.x = p2.x - 2
					ball.accelerate()
				}

				if ball.x <= 0 {
					p2.score++
					if p2.score >= winningScore {
						if winnerScreen(s, evCh, p1.score, p2.score, w, h, p2label, betOn) {
							p1.score, p2.score = 0, 0
							p1.vy, p2.vy = 0, 0
							ball.reset(cx, cy)
							serveTimer = serveWait
						} else {
							return
						}
					} else {
						scoreFlash(s, w, h, p2label)
						ball.reset(cx, cy)
						serveTimer = serveWait
					}
					scored = true
				}

				if !scored && ball.x >= float64(w-1) {
					p1.score++
					if p1.score >= winningScore {
						if winnerScreen(s, evCh, p1.score, p2.score, w, h, p1label, betOn) {
							p1.score, p2.score = 0, 0
							p1.vy, p2.vy = 0, 0
							ball.reset(cx, cy)
							serveTimer = serveWait
						} else {
							return
						}
					} else {
						scoreFlash(s, w, h, p1label)
						ball.reset(cx, cy)
						serveTimer = serveWait
					}
					scored = true
				}
			}
		}

		s.Clear()
		drawBorder(s, w, h)
		drawNet(s, w, h)
		drawHUD(s, p1.score, p2.score, w, p1label, p2label)

		if serveTimer > 0 {
			secs := int(math.Ceil(serveTimer))
			drawCentered(s, w, cy, fmt.Sprintf("%d", secs), styleWin)
		}

		p1.draw(s)
		p2.draw(s)
		ball.draw(s)

		var hint string
		switch mode {
		case 2:
			hint = " W/S:P1  ↑↓:P2  P:pause  R:restart  Q:quit "
		case 3:
			hint = fmt.Sprintf(" Airi vs Bo  (betting on: %s)  P:pause  R:restart  Q:quit ", betOn)
		default:
			hint = " W/S:P1  ↑↓:Airi  P:pause  R:restart  Q:quit "
		}
		drawCentered(s, w, h-1, hint, styleHint)

		s.Show()
	}
}

func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println()
	fmt.Println("  ╔══════════════════════════════╗")
	fmt.Println("  ║   PING PONG  (terminal)      ║")
	fmt.Println("  ╚══════════════════════════════╝")
	fmt.Println()
	fmt.Println("  Select mode:")
	fmt.Println("    1  —  1 Player  (vs AI Airi)")
	fmt.Println("    2  —  2 Players (local)")
	fmt.Println("    3  —  AI vs AI  (Airi vs Bo — place your bet!)")
	fmt.Println()

	var mode int
	for {
		fmt.Print("  Enter 1, 2, or 3: ")
		line, _ := reader.ReadString('\n')
		if len(line) > 0 {
			switch line[0] {
			case '1':
				mode = 1
			case '2':
				mode = 2
			case '3':
				mode = 3
			}
			if mode != 0 {
				break
			}
		}
		fmt.Println("  Please enter 1, 2, or 3.")
	}

	var betOn string
	if mode == 3 {
		fmt.Println()
		fmt.Println("  ┌─────────────────────────────────┐")
		fmt.Println("  │   🎲  Place your bet!            │")
		fmt.Println("  │   Who will win?                  │")
		fmt.Println("  │     A  —  Airi  (left paddle)    │")
		fmt.Println("  │     B  —  Bo    (right paddle)   │")
		fmt.Println("  └─────────────────────────────────┘")
		for {
			fmt.Print("  Enter A or B: ")
			line, _ := reader.ReadString('\n')
			if len(line) > 0 {
				switch line[0] {
				case 'a', 'A':
					betOn = "Airi"
				case 'b', 'B':
					betOn = "Bo"
				}
				if betOn != "" {
					break
				}
			}
			fmt.Println("  Please enter A or B.")
		}
		fmt.Printf("\n  💰  Bet placed on: %s — good luck!\n", betOn)
	}

	fmt.Println()
	fmt.Println("  Controls:")
	switch mode {
	case 1:
		fmt.Println("    W / S     — Player 1 (left paddle)")
		fmt.Println("    Airi      — controls right paddle")
	case 2:
		fmt.Println("    W / S     — Player 1 (left paddle)")
		fmt.Println("    ↑ / ↓     — Player 2 (right paddle)")
	case 3:
		fmt.Println("    Airi and Bo play automatically")
		fmt.Println("    Sit back and watch!")
	}
	fmt.Println("    P         — pause / resume")
	fmt.Println("    R         — restart match")
	fmt.Println("    Q / Esc   — quit")
	fmt.Printf("\n  First to %d points wins.\n\n", winningScore)
	fmt.Print("  Press Enter to start…")
	reader.ReadString('\n')

	s, err := tcell.NewScreen()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error creating screen:", err)
		os.Exit(1)
	}
	if err := s.Init(); err != nil {
		fmt.Fprintln(os.Stderr, "Error initialising screen:", err)
		os.Exit(1)
	}
	defer s.Fini()

	s.SetStyle(tcell.StyleDefault)
	s.EnableMouse()
	s.Clear()

	runGame(s, mode, betOn)

	fmt.Println("\n  Thanks for playing!\n")
}
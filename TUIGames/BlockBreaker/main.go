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
	fps            = 60
	paddleW        = 9
	ballSpeed0     = 0.65
	speedInc       = 0.012
	maxBallSpeed   = 2.0
	paddleMaxVel   = 2.4
	paddleAccel    = 0.46
	paddleFriction = 0.72
	keyHoldFrames  = 4

	blockRows = 5
	blockW    = 5
	topOffset = 4

	physSteps  = 4
	trailLen   = 6
	numSparks  = 5
	startLives = 3
	serveWait  = 2.0


	airiReaction = 0.91
	boReaction   = 0.80
	airiSpd      = 2.3
	boSpd        = 2.1
)

var frameDur = time.Second / fps


var (
	styleBall     = tcell.StyleDefault.Foreground(tcell.ColorWhite).Bold(true)
	stylePaddle   = tcell.StyleDefault.Foreground(tcell.ColorTeal).Bold(true)
	stylePaddleBo = tcell.StyleDefault.Foreground(tcell.ColorFuchsia).Bold(true)
	styleScore    = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	styleScoreBo  = tcell.StyleDefault.Foreground(tcell.ColorFuchsia).Bold(true)
	styleText     = tcell.StyleDefault.Foreground(tcell.ColorGreen).Bold(true)
	styleTrail    = tcell.StyleDefault.Foreground(tcell.ColorSilver)
	styleTrailBo  = tcell.StyleDefault.Foreground(tcell.NewHexColor(0xAA44AA))
	styleWin      = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	styleSpark    = tcell.StyleDefault.Foreground(tcell.ColorRed)
	styleSparkBo  = tcell.StyleDefault.Foreground(tcell.ColorFuchsia)
	styleBorder   = tcell.StyleDefault.Foreground(tcell.ColorNavy)
	styleDivider  = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x334455))
	styleHint     = tcell.StyleDefault.Foreground(tcell.ColorNavy)
	styleLives    = tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
	styleGameOver = tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
	styleBetWin   = tcell.StyleDefault.Foreground(tcell.ColorGreen).Bold(true)
	styleBetLoss  = tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
	styleDone     = tcell.StyleDefault.Foreground(tcell.ColorNavy)
)


var blockStyles = []tcell.Style{
	tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true),
	tcell.StyleDefault.Foreground(tcell.NewHexColor(0xFF6600)).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorGreen).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorAqua).Bold(true),
}

var blockGlyphs = []rune{'█', '▓', '▒', '░', '▪'}
var blockScores = []int{50, 40, 30, 20, 10}


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

func drawHUD(s tcell.Screen, score, lives, w int) {
	scoreStr := fmt.Sprintf(" ★ Score: %-6d", score)
	drawStr(s, 1, 0, scoreStr, styleScore)
	livesStr := " "
	for i := 0; i < lives; i++ {
		livesStr += "♥ "
	}
	runes := []rune(livesStr)
	drawStr(s, w-len(runes)-1, 0, livesStr, styleLives)
}

func centeredLines(s tcell.Screen, w, h int, lines []string, style tcell.Style) {
	sy := h/2 - len(lines)/2
	for i, line := range lines {
		drawCentered(s, w, sy+i, line, style)
	}
}

func clearFlash(s tcell.Screen, w, h int) {
	msg := "  ✦  CLEARED!  ✦  "
	for i := 0; i < 12; i++ {
		s.Clear()
		drawBorder(s, w, h)
		drawCentered(s, w, h/2, msg, styleWin)
		s.Show()
		time.Sleep(60 * time.Millisecond)
		s.Clear()
		drawBorder(s, w, h)
		s.Show()
		time.Sleep(40 * time.Millisecond)
	}
}


type Spark struct{ r, c, ttl int }
type Point struct{ r, c int }


type Block struct{ alive bool }

type Grid struct {
	blocks [][]Block
	cols   int
	topOff int
	xOff   int
}

func newGrid(xOff, availW, topOff int) *Grid {
	cols := availW / blockW
	if cols < 1 {
		cols = 1
	}
	g := &Grid{
		blocks: make([][]Block, blockRows),
		cols:   cols,
		topOff: topOff,
		xOff:   xOff,
	}
	for r := 0; r < blockRows; r++ {
		g.blocks[r] = make([]Block, cols)
		for c := 0; c < g.cols; c++ {
			g.blocks[r][c] = Block{alive: true}
		}
	}
	return g
}

func (g *Grid) remaining() int {
	n := 0
	for r := 0; r < blockRows; r++ {
		for c := 0; c < g.cols; c++ {
			if g.blocks[r][c].alive {
				n++
			}
		}
	}
	return n
}

func (g *Grid) draw(s tcell.Screen) {
	for r := 0; r < blockRows; r++ {
		ch := blockGlyphs[r]
		st := blockStyles[r]
		for c := 0; c < g.cols; c++ {
			if !g.blocks[r][c].alive {
				continue
			}
			bx := g.xOff + c*blockW
			by := g.topOff + r
			for i := 0; i < blockW-1; i++ {
				s.SetContent(bx+i, by, ch, nil, st)
			}
		}
	}
}

func (g *Grid) checkHit(b *Ball) int {
	bx := int(math.Round(b.x))
	by := int(math.Round(b.y))
	for r := 0; r < blockRows; r++ {
		for c := 0; c < g.cols; c++ {
			if !g.blocks[r][c].alive {
				continue
			}
			blx := g.xOff + c*blockW
			bly := g.topOff + r
			blr := blx + blockW - 2
			if bx >= blx && bx <= blr && by == bly {
				g.blocks[r][c].alive = false
				b.addSpark(bx, by)
				b.accelerate()
				if math.Abs(b.dy) >= math.Abs(b.dx) {
					b.dy = -b.dy
				} else {
					b.dx = -b.dx
				}
				b.dx += rand.Float64()*0.12 - 0.06
				return blockScores[r]
			}
		}
	}
	return 0
}


type Paddle struct {
	x, vx  float64
	y      int
	width  int
	xLeft  int
	xRight int
	style  tcell.Style
}

func newPaddle(cx, row, xLeft, xRight int, style tcell.Style) *Paddle {
	x := clamp(float64(cx-paddleW/2), float64(xLeft), float64(xRight-paddleW))
	return &Paddle{x: x, y: row, width: paddleW, xLeft: xLeft, xRight: xRight, style: style}
}

func (p *Paddle) left() int    { return int(p.x) }
func (p *Paddle) right() int   { return int(p.x) + p.width - 1 }
func (p *Paddle) mid() float64 { return p.x + float64(p.width)/2 }

func (p *Paddle) accelerate(dir float64) {
	p.vx += dir * paddleAccel
	p.vx = clamp(p.vx, -paddleMaxVel, paddleMaxVel)
	p.applyVel()
}

func (p *Paddle) coast() {
	p.vx *= paddleFriction
	if math.Abs(p.vx) < 0.05 {
		p.vx = 0
	}
	p.applyVel()
}

func (p *Paddle) aiMove(ballX, speed, reaction float64) {
	if rand.Float64() > reaction {
		return
	}
	diff := ballX - p.mid()
	if math.Abs(diff) < 1 {
		return
	}
	if diff < 0 {
		p.vx = -speed
	} else {
		p.vx = speed
	}
	p.applyVel()
}

func (p *Paddle) applyVel() {
	p.x += p.vx
	p.x = clamp(p.x, float64(p.xLeft), float64(p.xRight-p.width))
}

func (p *Paddle) draw(s tcell.Screen) {
	lx := p.left()
	for i := 0; i < p.width; i++ {
		ch := rune('━')
		if i == 0 || i == p.width-1 {
			ch = '▓'
		}
		s.SetContent(lx+i, p.y, ch, nil, p.style)
	}
}


type Ball struct {
	x, y, dx, dy, speed float64
	trail               []Point
	sparks              []Spark
	sball, strail, sspk tcell.Style
}

func newBall(cx, launchY int, sb, st, ss tcell.Style) *Ball {
	b := &Ball{sball: sb, strail: st, sspk: ss}
	b.reset(cx, launchY)
	return b
}

func (b *Ball) reset(cx, launchY int) {
	b.x = float64(cx)
	b.y = float64(launchY)
	angle := (rand.Float64()*0.5 - 0.25) * math.Pi
	b.dx = ballSpeed0 * math.Sin(angle)
	b.dy = -ballSpeed0 * math.Cos(angle)
	b.speed = ballSpeed0
	b.trail = b.trail[:0]
	b.sparks = b.sparks[:0]
}

func (b *Ball) addSpark(x, y int) {
	for i := 0; i < numSparks; i++ {
		b.sparks = append(b.sparks, Spark{r: y + rand.Intn(3) - 1, c: x + rand.Intn(3) - 1, ttl: 4})
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
		s.SetContent(sp.c, sp.r, '·', nil, b.sspk)
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
		s.SetContent(pt.c, pt.r, ch, nil, b.strail)
	}
	s.SetContent(int(b.x), int(b.y), '●', nil, b.sball)
}


func gameOverScreen(s tcell.Screen, evCh <-chan tcell.Event, score, w, h int) bool {
	lines := []string{
		"  ✗  GAME OVER  ✗  ",
		fmt.Sprintf("  Final Score:  %d  ", score),
		"",
		"  R: play again   Q: quit  ",
	}
	s.Clear()
	drawBorder(s, w, h)
	centeredLines(s, w, h, lines, styleGameOver)
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

func youWinScreen(s tcell.Screen, evCh <-chan tcell.Event, score, w, h int) bool {
	lines := []string{
		"  \U0001f3c6  YOU WIN!  \U0001f3c6  ",
		fmt.Sprintf("  Score:  %d  ", score),
		"",
		"  R: play again   Q: quit  ",
	}
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


func betResultScreen(s tcell.Screen, evCh <-chan tcell.Event,
	airiScore, boScore, w, h int, winner, betOn string) bool {

	var winnerLine string
	if winner == "TIE" {
		winnerLine = "  🤝  IT'S A TIE!  🤝  "
	} else {
		winnerLine = fmt.Sprintf("  \U0001f3c6  %s WINS!  \U0001f3c6  ", winner)
	}

	scoreBar := fmt.Sprintf("  Airi: %d   vs   Bo: %d  ", airiScore, boScore)

	var betLine string
	var betSt tcell.Style
	if betOn == "" {
		betLine = "  (no bet placed)  "
		betSt = styleHint
	} else if betOn == winner {
		betLine = fmt.Sprintf("  ✓  You bet on %s — WINNER!  💰  ", betOn)
		betSt = styleBetWin
	} else if winner == "TIE" {
		betLine = fmt.Sprintf("  ↔  You bet on %s — it's a tie, no winner.  ", betOn)
		betSt = styleHint
	} else {
		betLine = fmt.Sprintf("  ✗  You bet on %s — better luck next time.  ", betOn)
		betSt = styleBetLoss
	}

	lines := []string{
		winnerLine,
		scoreBar,
		"",
		betLine,
		"",
		"  R: replay   Q: quit  ",
	}

	s.Clear()
	drawBorder(s, w, h)
	sy := h/2 - len(lines)/2
	for i, line := range lines {
		st := styleWin
		if i == 3 {
			st = betSt
		}
		drawCentered(s, w, sy+i, line, st)
	}
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
	left, right          bool
	quit, pause, restart bool
}

type stickyKey struct {
	dir    float64
	frames int
}

func (sk *stickyKey) press(d float64) { sk.dir = d; sk.frames = keyHoldFrames }
func (sk *stickyKey) tick() float64 {
	if sk.frames <= 0 {
		return 0
	}
	sk.frames--
	return sk.dir
}


type HalfGame struct {
	grid       *Grid
	paddle     *Paddle
	ball       *Ball
	score      int
	lives      int
	serveTimer float64
	done       bool
	cleared    bool
	xLeft      int
	xRight     int
	cx         int
	paddleRow  int
	reaction   float64
	aiSpdVal   float64
}

func newHalfGame(xLeft, xRight, topOff, h int,
	paddleStyle, ballStyle, trailStyle, sparkStyle tcell.Style,
	reaction, aiSpdVal float64) *HalfGame {

	availW := xRight - xLeft
	cx := (xLeft + xRight) / 2
	prow := h - 3
	hg := &HalfGame{
		grid:       newGrid(xLeft, availW, topOff),
		paddle:     newPaddle(cx, prow, xLeft, xRight, paddleStyle),
		ball:       newBall(cx, prow-2, ballStyle, trailStyle, sparkStyle),
		lives:      startLives,
		serveTimer: serveWait,
		xLeft:      xLeft,
		xRight:     xRight,
		cx:         cx,
		paddleRow:  prow,
		reaction:   reaction,
		aiSpdVal:   aiSpdVal,
	}
	return hg
}

func (hg *HalfGame) resize(h int) {
	hg.paddleRow = h - 3
	hg.paddle.y = hg.paddleRow
}

func (hg *HalfGame) resetRound() {
	hg.paddle.x = float64(hg.cx - paddleW/2)
	hg.paddle.vx = 0
	hg.ball.reset(hg.cx, hg.paddleRow-2)
	hg.serveTimer = serveWait
}


func (hg *HalfGame) step(h int) {
	if hg.done || hg.cleared {
		return
	}
	hg.paddle.aiMove(hg.ball.x, hg.aiSpdVal, hg.reaction)

	if hg.serveTimer > 0 {
		hg.serveTimer -= frameDur.Seconds()
		hg.ball.x = hg.paddle.mid()
		hg.ball.y = float64(hg.paddleRow - 2)
		return
	}

	hg.ball.updateTrail()
	lostBall := false

	for step := 0; step < physSteps && !lostBall && !hg.cleared; step++ {
		hg.ball.x += hg.ball.dx / physSteps
		hg.ball.y += hg.ball.dy / physSteps
		bx := int(hg.ball.x)
		by := int(hg.ball.y)

		if hg.ball.x <= float64(hg.xLeft) {
			hg.ball.x = float64(hg.xLeft)
			hg.ball.dx = math.Abs(hg.ball.dx)
		}
		if hg.ball.x >= float64(hg.xRight-1) {
			hg.ball.x = float64(hg.xRight - 1)
			hg.ball.dx = -math.Abs(hg.ball.dx)
		}
		if hg.ball.y <= 1 {
			hg.ball.y = 1
			hg.ball.dy = math.Abs(hg.ball.dy)
		}

		if hg.ball.dy > 0 && by == hg.paddleRow &&
			bx >= hg.paddle.left() && bx <= hg.paddle.right() {
			hg.ball.addSpark(bx, by)
			rel := (hg.ball.x - hg.paddle.mid()) / (float64(hg.paddle.width) / 2)
			rel = clamp(rel, -0.98, 0.98)
			angle := rel * math.Pi * 0.40
			spd := math.Hypot(hg.ball.dx, hg.ball.dy)
			hg.ball.dx = spd * math.Sin(angle)
			hg.ball.dy = -math.Abs(spd * math.Cos(angle))
			hg.ball.y = float64(hg.paddleRow - 1)
		}

		hg.score += hg.grid.checkHit(hg.ball)

		if hg.ball.y >= float64(h-2) {
			lostBall = true
		}
		if hg.grid.remaining() == 0 {
			hg.score += 1000
			hg.cleared = true
		}
	}

	if lostBall {
		hg.ball.addSpark(int(hg.ball.x), h-2)
		hg.lives--
		if hg.lives <= 0 {
			hg.done = true
		} else {
			hg.resetRound()
		}
	}
}

func (hg *HalfGame) draw(s tcell.Screen, h int) {
	hg.grid.draw(s)
	if !hg.done {
		hg.paddle.draw(s)

		livesStr := ""
		for i := 0; i < hg.lives; i++ {
			livesStr += "♥ "
		}
		drawStr(s, hg.xLeft, hg.paddleRow-1, livesStr, styleLives)
	}
	if !hg.done && !hg.cleared {
		hg.ball.draw(s)
	}
}


func drawDualHUD(s tcell.Screen, airi, bo *HalfGame, w int, betOn string) {
	airiStr := fmt.Sprintf(" Airi ★ %d ", airi.score)
	drawStr(s, 1, 0, airiStr, styleScore)

	boStr := fmt.Sprintf(" Bo ★ %d ", bo.score)
	boRunes := []rune(boStr)
	drawStr(s, w-1-len(boRunes), 0, boStr, styleScoreBo)

	if betOn != "" {
		betInd := fmt.Sprintf(" 🎲 bet: %s ", betOn)
		drawCentered(s, w, 0, betInd, styleBetWin)
	}
}

func drawDivider(s tcell.Screen, midX, h int) {
	for y := 1; y < h-1; y++ {
		ch := rune(' ')
		if y%2 == 0 {
			ch = '┊'
		}
		s.SetContent(midX, y, ch, nil, styleDivider)
	}
}


func runGame(s tcell.Screen, mode int) {
	w, h := s.Size()
	topOff := topOffset + 1

	paddleRow := func() int { return h - 3 }
	launchY := func() int { return paddleRow() - 2 }
	cx := func() int { return w / 2 }

	var (
		paddle     = newPaddle(cx(), paddleRow(), 1, w-1, stylePaddle)
		ball       = newBall(cx(), launchY(), styleBall, styleTrail, styleSpark)
		grid       = newGrid(1, w-2, topOff)
		score      = 0
		lives      = startLives
		serveTimer = serveWait
		paused     = false
		sk         stickyKey
	)

	resetRound := func() {
		paddle.x = float64(cx() - paddleW/2)
		paddle.vx = 0
		ball.reset(cx(), launchY())
		serveTimer = serveWait
	}
	resetGame := func() {
		grid = newGrid(1, w-2, topOff)
		score = 0
		lives = startLives
		resetRound()
	}

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
		paddle.y = paddleRow()
		paddle.xRight = w - 1

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
					resetGame()
				case *tcell.EventKey:
					switch {
					case e.Key() == tcell.KeyEscape || e.Rune() == 'q' || e.Rune() == 'Q':
						inp.quit = true
					case e.Rune() == 'p' || e.Rune() == 'P':
						inp.pause = true
					case e.Rune() == 'r' || e.Rune() == 'R':
						inp.restart = true
					case e.Rune() == 'a' || e.Rune() == 'A' || e.Key() == tcell.KeyLeft:
						sk.press(-1)
					case e.Rune() == 'd' || e.Rune() == 'D' || e.Key() == tcell.KeyRight:
						sk.press(1)
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
			resetGame()
		}

		if paused {
			s.Clear()
			drawBorder(s, w, h)
			centeredLines(s, w, h, []string{"── PAUSED ──", "P to resume"}, styleText)
			s.Show()
			continue
		}

		if mode == 2 {
			paddle.aiMove(ball.x, airiSpd, airiReaction)
		} else {
			d := sk.tick()
			if d != 0 {
				paddle.accelerate(d)
			} else {
				paddle.coast()
			}
		}

		if serveTimer > 0 {
			serveTimer -= frameDur.Seconds()
			ball.x = paddle.mid()
			ball.y = float64(paddleRow() - 2)
		} else {
			ball.updateTrail()
			lostBall := false
			cleared := false

			for step := 0; step < physSteps && !lostBall && !cleared; step++ {
				ball.x += ball.dx / physSteps
				ball.y += ball.dy / physSteps
				bx := int(ball.x)
				by := int(ball.y)

				if ball.x <= 1 {
					ball.x = 1
					ball.dx = math.Abs(ball.dx)
				}
				if ball.x >= float64(w-2) {
					ball.x = float64(w - 2)
					ball.dx = -math.Abs(ball.dx)
				}
				if ball.y <= 1 {
					ball.y = 1
					ball.dy = math.Abs(ball.dy)
				}

				if ball.dy > 0 && by == paddle.y && bx >= paddle.left() && bx <= paddle.right() {
					ball.addSpark(bx, by)
					rel := (ball.x - paddle.mid()) / (float64(paddle.width) / 2)
					rel = clamp(rel, -0.98, 0.98)
					angle := rel * math.Pi * 0.40
					spd := math.Hypot(ball.dx, ball.dy)
					ball.dx = spd * math.Sin(angle)
					ball.dy = -math.Abs(spd * math.Cos(angle))
					ball.y = float64(paddle.y - 1)
				}

				score += grid.checkHit(ball)

				if ball.y >= float64(h-2) {
					lostBall = true
				}
				if grid.remaining() == 0 {
					cleared = true
				}
			}

			if lostBall {
				ball.addSpark(int(ball.x), h-2)
				lives--
				if lives <= 0 {
					s.Clear()
					drawBorder(s, w, h)
					drawHUD(s, score, 0, w)
					s.Show()
					if gameOverScreen(s, evCh, score, w, h) {
						resetGame()
					} else {
						return
					}
				} else {
					resetRound()
				}
				continue
			}

			if cleared {
				score += 1000
				clearFlash(s, w, h)
				if youWinScreen(s, evCh, score, w, h) {
					resetGame()
				} else {
					return
				}
				continue
			}
		}

		s.Clear()
		drawBorder(s, w, h)
		drawHUD(s, score, lives, w)
		grid.draw(s)
		paddle.draw(s)
		ball.draw(s)

		if serveTimer > 0 {
			secs := int(math.Ceil(serveTimer))
			drawCentered(s, w, paddleRow()-4, fmt.Sprintf("  %d  ", secs), styleWin)
		}

		var hint string
		if mode == 2 {
			hint = " AI:Airi  P:pause  R:restart  Q:quit "
		} else {
			hint = " A/←:left  D/→:right  P:pause  R:restart  Q:quit "
		}
		drawCentered(s, w, h-1, hint, styleHint)
		s.Show()
	}
}


func runDualGame(s tcell.Screen, betOn string) {
	w, h := s.Size()
	topOff := topOffset + 1

	buildHalves := func() (*HalfGame, *HalfGame) {
		midX := w / 2
		airi := newHalfGame(1, midX, topOff, h,
			stylePaddle, styleBall, styleTrail, styleSpark,
			airiReaction, airiSpd)
		bo := newHalfGame(midX+1, w-1, topOff, h,
			stylePaddleBo, styleBall, styleTrailBo, styleSparkBo,
			boReaction, boSpd)
		return airi, bo
	}

	airi, bo := buildHalves()

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

	paused := false
	ticker := time.NewTicker(frameDur)
	defer ticker.Stop()

	for range ticker.C {
		w, h = s.Size()
		midX := w / 2

		airi.resize(h)
		bo.resize(h)

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
					airi, bo = buildHalves()
				case *tcell.EventKey:
					switch {
					case e.Key() == tcell.KeyEscape || e.Rune() == 'q' || e.Rune() == 'Q':
						inp.quit = true
					case e.Rune() == 'p' || e.Rune() == 'P':
						inp.pause = true
					case e.Rune() == 'r' || e.Rune() == 'R':
						inp.restart = true
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
			airi, bo = buildHalves()
		}

		if paused {
			s.Clear()
			drawBorder(s, w, h)
			drawDivider(s, midX, h)
			centeredLines(s, w, h, []string{"── PAUSED ──", "P to resume"}, styleText)
			s.Show()
			continue
		}

		airi.step(h)
		bo.step(h)


		airiFinished := airi.done || airi.cleared
		boFinished := bo.done || bo.cleared

		if airiFinished || boFinished {
			var winner string
			switch {
			case airi.cleared && !bo.cleared:
				winner = "Airi"
			case bo.cleared && !airi.cleared:
				winner = "Bo"
			case airi.cleared && bo.cleared:
				if airi.score > bo.score {
					winner = "Airi"
				} else if bo.score > airi.score {
					winner = "Bo"
				} else {
					winner = "TIE"
				}
			case airi.done && !bo.done:
				winner = "Bo"
			case bo.done && !airi.done:
				winner = "Airi"
			default:
				if airi.score > bo.score {
					winner = "Airi"
				} else if bo.score > airi.score {
					winner = "Bo"
				} else {
					winner = "TIE"
				}
			}


			winMsg := fmt.Sprintf("  \U0001f3c6  %s WINS!  \U0001f3c6  ", winner)
			if winner == "TIE" {
				winMsg = "  🤝  IT'S A TIE!  🤝  "
			}
			for i := 0; i < 16; i++ {
				s.Clear()
				drawBorder(s, w, h)
				drawDivider(s, midX, h)
				drawCentered(s, w, h/2, winMsg, styleWin)
				s.Show()
				time.Sleep(65 * time.Millisecond)
				s.Clear()
				drawBorder(s, w, h)
				drawDivider(s, midX, h)
				s.Show()
				time.Sleep(40 * time.Millisecond)
			}

			if betResultScreen(s, evCh, airi.score, bo.score, w, h, winner, betOn) {
				airi, bo = buildHalves()
			} else {
				return
			}
			continue
		}


		s.Clear()
		drawBorder(s, w, h)
		drawDivider(s, midX, h)
		drawDualHUD(s, airi, bo, w, betOn)

		airi.draw(s, h)
		bo.draw(s, h)


		if airi.done {
			drawCentered(s, midX, h/2, " ✗ BUST ✗ ", styleDone)
		}
		if bo.done {
			boCenterX := midX + (w-midX)/2
			drawStr(s, boCenterX-5, h/2, " ✗ BUST ✗ ", styleDone)
		}


		if !airi.done && !airi.cleared && airi.serveTimer > 0 {
			secs := int(math.Ceil(airi.serveTimer))
			drawCentered(s, midX, airi.paddleRow-4, fmt.Sprintf(" %d ", secs), styleWin)
		}
		if !bo.done && !bo.cleared && bo.serveTimer > 0 {
			secs := int(math.Ceil(bo.serveTimer))
			boCenterX := midX + (w-midX)/2
			drawStr(s, boCenterX, bo.paddleRow-4, fmt.Sprintf(" %d ", secs), styleWin)
		}

		betHint := ""
		if betOn != "" {
			betHint = fmt.Sprintf(" 🎲%s ", betOn)
		}
		drawCentered(s, w, h-1, betHint+"P:pause  R:restart  Q:quit", styleHint)
		s.Show()
	}
}


func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println()
	fmt.Println("  ╔══════════════════════════════╗")
	fmt.Println("  ║   BLOCK BREAKER (terminal)   ║")
	fmt.Println("  ╚══════════════════════════════╝")
	fmt.Println()
	fmt.Println("  Select mode:")
	fmt.Println("    1  —  Human      (you control the paddle)")
	fmt.Println("    2  —  AI         (watch Airi play)")
	fmt.Println("    3  —  AI vs AI   (Airi vs Bo — place your bet!)")
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
		fmt.Println("  ┌───────────────────────────────────────┐")
		fmt.Println("  │   🎲  Place your bet!                  │")
		fmt.Println("  │   Who clears their board first?        │")
		fmt.Println("  │     A  —  Airi  (left  side, teal)    │")
		fmt.Println("  │     B  —  Bo    (right side, purple)   │")
		fmt.Println("  │   (Press Enter to skip betting)        │")
		fmt.Println("  └───────────────────────────────────────┘")
		fmt.Print("  Enter A or B: ")
		line, _ := reader.ReadString('\n')
		if len(line) > 0 {
			switch line[0] {
			case 'a', 'A':
				betOn = "Airi"
			case 'b', 'B':
				betOn = "Bo"
			}
		}
		if betOn != "" {
			fmt.Printf("\n  💰  Bet placed on: %s — good luck!\n", betOn)
		} else {
			fmt.Println("\n  No bet placed. Enjoy the show!")
		}
	}

	fmt.Println()
	fmt.Println("  Controls:")
	switch mode {
	case 1:
		fmt.Println("    A / ←    — move paddle left")
		fmt.Println("    D / →    — move paddle right")
	case 2:
		fmt.Println("    Airi controls the paddle — sit back!")
	case 3:
		fmt.Println("    Split screen: Airi (teal, left)  vs  Bo (purple, right)")
		fmt.Println("    Both AIs play autonomously on their own half.")
	}
	fmt.Println("    P        — pause / resume")
	fmt.Println("    R        — restart")
	fmt.Println("    Q / Esc  — quit")
	fmt.Println()
	fmt.Printf("  Block scores: RED=50  ORANGE=40  YELLOW=30  GREEN=20  CYAN=10\n")
	if mode == 3 {
		fmt.Println("  First to clear their half wins (+ 1000 bonus).")
		fmt.Println("  If both bust, highest score wins.")
		fmt.Println("  Personalities:  Airi = precise (91%)  |  Bo = erratic (80%)")
	} else {
		fmt.Println("  Clear the board for a 1000-point bonus!")
		fmt.Println("  You have 3 lives.")
	}
	fmt.Print("\n  Press Enter to start…")
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

	if mode == 3 {
		runDualGame(s, betOn)
	} else {
		runGame(s, mode)
	}

	fmt.Println("\n  Thanks for playing!\n")
}
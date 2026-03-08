package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/gdamore/tcell/v2"
)


const (
	boardW = 10
	boardH = 20
	fps    = 60
)

var frameDur = time.Second / fps


var (
	styleBorder   = tcell.StyleDefault.Foreground(tcell.ColorNavy)
	styleLabel    = tcell.StyleDefault.Foreground(tcell.ColorSilver)
	styleScore    = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	styleHint     = tcell.StyleDefault.Foreground(tcell.ColorNavy)
	styleText     = tcell.StyleDefault.Foreground(tcell.ColorGreen).Bold(true)
	styleGameOver = tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
	styleGhost    = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x1a1a3a))
	styleEmpty    = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x0d0d20))
)


var pieceStyles = [7]tcell.Style{
	tcell.StyleDefault.Foreground(tcell.ColorAqua).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorFuchsia).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorGreen).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorBlue).Bold(true),
	tcell.StyleDefault.Foreground(tcell.NewHexColor(0xFF6600)).Bold(true),
}




var pieceData = [7][4][4][2]int{

	{
		{{1, 0}, {1, 1}, {1, 2}, {1, 3}},
		{{0, 2}, {1, 2}, {2, 2}, {3, 2}},
		{{2, 0}, {2, 1}, {2, 2}, {2, 3}},
		{{0, 1}, {1, 1}, {2, 1}, {3, 1}},
	},

	{
		{{0, 1}, {0, 2}, {1, 1}, {1, 2}},
		{{0, 1}, {0, 2}, {1, 1}, {1, 2}},
		{{0, 1}, {0, 2}, {1, 1}, {1, 2}},
		{{0, 1}, {0, 2}, {1, 1}, {1, 2}},
	},

	{
		{{0, 1}, {1, 0}, {1, 1}, {1, 2}},
		{{0, 1}, {1, 1}, {1, 2}, {2, 1}},
		{{1, 0}, {1, 1}, {1, 2}, {2, 1}},
		{{0, 1}, {1, 0}, {1, 1}, {2, 1}},
	},

	{
		{{0, 1}, {0, 2}, {1, 0}, {1, 1}},
		{{0, 0}, {1, 0}, {1, 1}, {2, 1}},
		{{0, 1}, {0, 2}, {1, 0}, {1, 1}},
		{{0, 0}, {1, 0}, {1, 1}, {2, 1}},
	},

	{
		{{0, 0}, {0, 1}, {1, 1}, {1, 2}},
		{{0, 1}, {1, 0}, {1, 1}, {2, 0}},
		{{0, 0}, {0, 1}, {1, 1}, {1, 2}},
		{{0, 1}, {1, 0}, {1, 1}, {2, 0}},
	},

	{
		{{0, 0}, {1, 0}, {1, 1}, {1, 2}},
		{{0, 1}, {0, 2}, {1, 1}, {2, 1}},
		{{1, 0}, {1, 1}, {1, 2}, {2, 2}},
		{{0, 1}, {1, 1}, {2, 0}, {2, 1}},
	},

	{
		{{0, 2}, {1, 0}, {1, 1}, {1, 2}},
		{{0, 1}, {1, 1}, {2, 1}, {2, 2}},
		{{1, 0}, {1, 1}, {1, 2}, {2, 0}},
		{{0, 0}, {0, 1}, {1, 1}, {2, 1}},
	},
}


type bag struct {
	pool [7]int
	idx  int
}

func newBag() *bag {
	b := &bag{}
	for i := range b.pool {
		b.pool[i] = i
	}
	rand.Shuffle(7, func(i, j int) { b.pool[i], b.pool[j] = b.pool[j], b.pool[i] })
	return b
}

func (b *bag) next() int {
	if b.idx >= 7 {
		rand.Shuffle(7, func(i, j int) { b.pool[i], b.pool[j] = b.pool[j], b.pool[i] })
		b.idx = 0
	}
	p := b.pool[b.idx]
	b.idx++
	return p
}


type Game struct {
	board [boardH][boardW]int

	curType int
	curRot  int
	curRow  int
	curCol  int

	nextType int
	holdType int
	canHold  bool

	score int
	level int
	lines int

	over   bool
	paused bool

	dropTimer  float64
	lockActive bool
	lockDelay  int

	rng *bag
}

func newGame() *Game {
	g := &Game{holdType: -1, canHold: true, level: 1, rng: newBag()}
	for r := range g.board {
		for c := range g.board[r] {
			g.board[r][c] = -1
		}
	}
	g.nextType = g.rng.next()
	g.spawnNext()
	return g
}



func (g *Game) dropFrames() float64 {
	ms := 1000 - (g.level-1)*70
	if ms < 50 {
		ms = 50
	}
	return float64(ms) * float64(fps) / 1000.0
}

func (g *Game) fits(t, rot, row, col int) bool {
	for _, cell := range pieceData[t][rot] {
		r, c := row+cell[0], col+cell[1]
		if r >= boardH || c < 0 || c >= boardW {
			return false
		}
		if r >= 0 && g.board[r][c] != -1 {
			return false
		}
	}
	return true
}

func (g *Game) spawnNext() {
	g.curType = g.nextType
	g.curRot = 0
	g.curRow = -1
	g.curCol = 3
	g.nextType = g.rng.next()
	g.canHold = true
	g.lockActive = false
	g.dropTimer = g.dropFrames()
	if !g.fits(g.curType, g.curRot, g.curRow, g.curCol) {
		g.over = true
	}
}

func (g *Game) ghostRow() int {
	r := g.curRow
	for g.fits(g.curType, g.curRot, r+1, g.curCol) {
		r++
	}
	return r
}

func (g *Game) lock() {
	for _, cell := range pieceData[g.curType][g.curRot] {
		r, c := g.curRow+cell[0], g.curCol+cell[1]
		if r >= 0 {
			g.board[r][c] = g.curType
		}
	}
	g.clearLines()
	if !g.over {
		g.spawnNext()
	}
}

func (g *Game) clearLines() {
	lineScores := [5]int{0, 100, 300, 500, 800}
	cleared := 0
	writeRow := boardH - 1
	var newBoard [boardH][boardW]int
	for r := range newBoard {
		for c := range newBoard[r] {
			newBoard[r][c] = -1
		}
	}
	for r := boardH - 1; r >= 0; r-- {
		full := true
		for c := 0; c < boardW; c++ {
			if g.board[r][c] == -1 {
				full = false
				break
			}
		}
		if !full {
			newBoard[writeRow] = g.board[r]
			writeRow--
		} else {
			cleared++
		}
	}
	g.board = newBoard
	if cleared > 0 && cleared <= 4 {
		g.score += lineScores[cleared] * g.level
		g.lines += cleared
		g.level = g.lines/10 + 1
		if g.level > 15 {
			g.level = 15
		}
	}
}

func (g *Game) step(soft bool) {
	canFall := g.fits(g.curType, g.curRot, g.curRow+1, g.curCol)
	if !canFall {
		if !g.lockActive {
			g.lockActive = true
			g.lockDelay = 30
		} else {
			g.lockDelay--
			if g.lockDelay <= 0 {
				g.lock()
			}
		}
		return
	}
	g.lockActive = false
	interval := g.dropFrames()
	if soft {
		interval = 2
	}
	g.dropTimer--
	if g.dropTimer <= 0 {
		g.dropTimer = interval
		g.curRow++
		if soft {
			g.score++
		}
	}
}

func (g *Game) moveLeft() {
	if g.fits(g.curType, g.curRot, g.curRow, g.curCol-1) {
		g.curCol--
		g.lockActive = false
	}
}

func (g *Game) moveRight() {
	if g.fits(g.curType, g.curRot, g.curRow, g.curCol+1) {
		g.curCol++
		g.lockActive = false
	}
}

func (g *Game) rotate(dir int) {
	newRot := (g.curRot + dir + 4) % 4
	kicks := [][2]int{{0, 0}, {0, -1}, {0, 1}, {0, -2}, {0, 2}, {-1, 0}}
	for _, k := range kicks {
		if g.fits(g.curType, newRot, g.curRow+k[0], g.curCol+k[1]) {
			g.curRot = newRot
			g.curRow += k[0]
			g.curCol += k[1]
			g.lockActive = false
			return
		}
	}
}

func (g *Game) hardDrop() {
	dist := 0
	for g.fits(g.curType, g.curRot, g.curRow+1, g.curCol) {
		g.curRow++
		dist++
	}
	g.score += dist * 2
	g.lock()
}

func (g *Game) hold() {
	if !g.canHold {
		return
	}
	g.canHold = false
	if g.holdType == -1 {
		g.holdType = g.curType
		g.spawnNext()
	} else {
		g.holdType, g.curType = g.curType, g.holdType
		g.curRot = 0
		g.curRow = -1
		g.curCol = 3
		g.lockActive = false
		g.dropTimer = g.dropFrames()
		if !g.fits(g.curType, g.curRot, g.curRow, g.curCol) {
			g.over = true
		}
	}
}


func drawStr(s tcell.Screen, x, y int, text string, style tcell.Style) {
	for i, ch := range []rune(text) {
		s.SetContent(x+i, y, ch, nil, style)
	}
}

func drawBorder(s tcell.Screen, x, y, w, h int, st tcell.Style) {
	for i := 1; i < w-1; i++ {
		s.SetContent(x+i, y, tcell.RuneHLine, nil, st)
		s.SetContent(x+i, y+h-1, tcell.RuneHLine, nil, st)
	}
	for i := 1; i < h-1; i++ {
		s.SetContent(x, y+i, tcell.RuneVLine, nil, st)
		s.SetContent(x+w-1, y+i, tcell.RuneVLine, nil, st)
	}
	s.SetContent(x, y, tcell.RuneULCorner, nil, st)
	s.SetContent(x+w-1, y, tcell.RuneURCorner, nil, st)
	s.SetContent(x, y+h-1, tcell.RuneLLCorner, nil, st)
	s.SetContent(x+w-1, y+h-1, tcell.RuneLRCorner, nil, st)
}

func setBlock(s tcell.Screen, x, y int, st tcell.Style) {
	s.SetContent(x, y, '█', nil, st)
	s.SetContent(x+1, y, '█', nil, st)
}


func (g *Game) draw(s tcell.Screen, bx, by int) {

	for r := 0; r < boardH; r++ {
		for c := 0; c < boardW; c++ {
			sx, sy := bx+c*2, by+r
			if g.board[r][c] >= 0 {
				setBlock(s, sx, sy, pieceStyles[g.board[r][c]])
			} else {
				s.SetContent(sx, sy, '·', nil, styleEmpty)
				s.SetContent(sx+1, sy, ' ', nil, styleEmpty)
			}
		}
	}

	ghost := g.ghostRow()


	if ghost > g.curRow {
		for _, cell := range pieceData[g.curType][g.curRot] {
			r, c := ghost+cell[0], g.curCol+cell[1]
			if r >= 0 && r < boardH && g.board[r][c] == -1 {
				s.SetContent(bx+c*2, by+r, '░', nil, styleGhost)
				s.SetContent(bx+c*2+1, by+r, '░', nil, styleGhost)
			}
		}
	}


	for _, cell := range pieceData[g.curType][g.curRot] {
		r, c := g.curRow+cell[0], g.curCol+cell[1]
		if r >= 0 && r < boardH {
			setBlock(s, bx+c*2, by+r, pieceStyles[g.curType])
		}
	}
}


func drawPreview(s tcell.Screen, px, py, pType int, dim bool) {
	if pType < 0 {
		return
	}
	cells := pieceData[pType][0]
	st := pieceStyles[pType]
	if dim {
		st = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x2a2a2a))
	}

	minR, minC, maxR, maxC := 9, 9, 0, 0
	for _, cell := range cells {
		if cell[0] < minR {
			minR = cell[0]
		}
		if cell[1] < minC {
			minC = cell[1]
		}
		if cell[0] > maxR {
			maxR = cell[0]
		}
		if cell[1] > maxC {
			maxC = cell[1]
		}
	}
	h, w := maxR-minR+1, maxC-minC+1
	offR := (2 - h) / 2
	offC := (4 - w) / 2

	for _, cell := range cells {
		r := cell[0] - minR + offR
		c := cell[1] - minC + offC
		if r >= 0 && c >= 0 {
			setBlock(s, px+c*2, py+r, st)
		}
	}
}

func (g *Game) drawPanel(s tcell.Screen, px, py int) {
	drawStr(s, px, py+0, "SCORE", styleLabel)
	drawStr(s, px, py+1, fmt.Sprintf("%-8d", g.score), styleScore)

	drawStr(s, px, py+3, "LEVEL", styleLabel)
	drawStr(s, px, py+4, fmt.Sprintf("%-2d", g.level), styleScore)

	drawStr(s, px, py+6, "LINES", styleLabel)
	drawStr(s, px, py+7, fmt.Sprintf("%-4d", g.lines), styleScore)


	drawStr(s, px, py+9, "NEXT", styleLabel)
	drawBorder(s, px-1, py+10, 10, 4, styleBorder)
	drawPreview(s, px, py+11, g.nextType, false)


	drawStr(s, px, py+15, "HOLD", styleLabel)
	if !g.canHold {
		drawStr(s, px+5, py+15, " ✗", styleGameOver)
	}
	drawBorder(s, px-1, py+16, 10, 4, styleBorder)
	drawPreview(s, px, py+17, g.holdType, !g.canHold)
}


func runTetris(s tcell.Screen) {
	g := newGame()

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

	softDrop := false

	for range ticker.C {
		tw, th := s.Size()

		const boardVisW = boardW*2 + 2
		const boardVisH = boardH + 2
		const panelW = 12
		const totalW = boardVisW + 2 + panelW

		bx := (tw - totalW) / 2
		by := (th - boardVisH) / 2
		if bx < 0 {
			bx = 0
		}
		if by < 0 {
			by = 0
		}
		px := bx + boardVisW + 2

		softDrop = false

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
				case *tcell.EventKey:
					k, r := e.Key(), e.Rune()


					if g.over {
						switch {
						case r == 'r' || r == 'R':
							g = newGame()
						case k == tcell.KeyEscape || r == 'q' || r == 'Q':
							return
						}
						break
					}


					if g.paused {
						switch {
						case r == 'p' || r == 'P':
							g.paused = false
						case k == tcell.KeyEscape || r == 'q' || r == 'Q':
							return
						}
						break
					}


					switch {
					case k == tcell.KeyEscape || r == 'q' || r == 'Q':
						return
					case r == 'p' || r == 'P':
						g.paused = true
					case r == 'r' || r == 'R':
						g = newGame()
					case k == tcell.KeyLeft || r == 'a' || r == 'A':
						g.moveLeft()
					case k == tcell.KeyRight || r == 'd' || r == 'D':
						g.moveRight()
					case k == tcell.KeyUp || r == 'w' || r == 'W' || r == 'x' || r == 'X':
						g.rotate(1)
					case r == 'z' || r == 'Z':
						g.rotate(-1)
					case k == tcell.KeyDown || r == 's' || r == 'S':
						softDrop = true
					case r == ' ':
						g.hardDrop()
					case r == 'c' || r == 'C':
						g.hold()
					}
				}
			default:
				break drainLoop
			}
		}

		if !g.over && !g.paused {
			g.step(softDrop)
		}


		s.Clear()
		drawBorder(s, bx, by, boardVisW, boardVisH, styleBorder)

		if g.paused {
			mid := by + boardVisH/2
			drawStr(s, bx+5, mid-1, "── PAUSED ──", styleText)
			drawStr(s, bx+6, mid+1, "P to resume", styleHint)
		} else {
			g.draw(s, bx+1, by+1)
		}

		g.drawPanel(s, px, by)


		if g.over {
			mid := by + boardVisH/2
			cw := boardW * 2
			cx := bx + 1

			for row := mid - 3; row <= mid+3; row++ {
				for col := cx; col < cx+cw; col++ {
					s.SetContent(col, row, ' ', nil,
						tcell.StyleDefault.Background(tcell.ColorBlack))
				}
			}
			goText := " ─ GAME OVER ─ "
			scoreText := fmt.Sprintf("  Score: %6d", g.score)
			hintText := " R:restart  Q:quit "
			drawStr(s, cx+(cw-len([]rune(goText)))/2, mid-1, goText, styleGameOver)
			drawStr(s, cx+(cw-len([]rune(scoreText)))/2, mid+1, scoreText, styleScore)
			drawStr(s, cx+(cw-len([]rune(hintText)))/2, mid+3, hintText, styleHint)
		}


		hint := " ←/A:left  D/→:right  ↑/W:spin↻  Z:spin↺  ↓/S:soft  SPC:drop  C:hold  P:pause  Q:quit "
		if len([]rune(hint)) < tw {
			drawStr(s, (tw-len([]rune(hint)))/2, th-1, hint, styleHint)
		}

		s.Show()
	}
}


func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println()
	fmt.Println("  ╔══════════════════════════════╗")
	fmt.Println("  ║      TETRIS  (terminal)      ║")
	fmt.Println("  ╚══════════════════════════════╝")
	fmt.Println()
	fmt.Println("  Controls:")
	fmt.Println("    ←/A · D/→       move left / right")
	fmt.Println("    ↑/W · Z         rotate CW / CCW")
	fmt.Println("    ↓/S             soft drop  (+1 pt / cell)")
	fmt.Println("    Space           hard drop  (+2 pt / cell)")
	fmt.Println("    C               hold piece")
	fmt.Println("    P               pause / resume")
	fmt.Println("    R               restart")
	fmt.Println("    Q / Esc         quit")
	fmt.Println()
	fmt.Println("  Scoring  (× level):")
	fmt.Println("    1 line = 100    2 lines = 300")
	fmt.Println("    3 lines = 500   4 lines = 800  ✦ TETRIS!")
	fmt.Println()
	fmt.Print("  Press Enter to start…")
	reader.ReadString('\n')

	s, err := tcell.NewScreen()
	if err != nil {
		fmt.Fprintln(os.Stderr, "screen error:", err)
		os.Exit(1)
	}
	if err := s.Init(); err != nil {
		fmt.Fprintln(os.Stderr, "init error:", err)
		os.Exit(1)
	}
	defer s.Fini()
	s.SetStyle(tcell.StyleDefault)
	s.Clear()

	runTetris(s)
	fmt.Println("\n  Thanks for playing!\n")
}
package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/notnil/chess"
)


const (
	bX    = 4
	bY    = 3
	cellW = 4
	cellH = 2
	iX    = 42
)


type Diff struct {
	label string
	skill int
	depth int
}

var diffs = []Diff{
	{"Easy", 2, 3},
	{"Medium", 10, 8},
	{"Hard", 20, 15},
}


type SF struct {
	cmd  *exec.Cmd
	in   io.WriteCloser
	scan *bufio.Scanner
}

func newSF(d Diff) (*SF, error) {
	path := findSF()
	if path == "" {
		return nil, fmt.Errorf("stockfish not found — install with: sudo apt install stockfish")
	}
	cmd := exec.Command(path)
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("could not start stockfish: %w", err)
	}
	sf := &SF{cmd: cmd, in: in, scan: bufio.NewScanner(out)}
	sf.tx("uci")
	sf.expect("uciok")
	sf.tx(fmt.Sprintf("setoption name Skill Level value %d", d.skill))
	sf.tx("ucinewgame")
	sf.tx("isready")
	sf.expect("readyok")
	return sf, nil
}

func findSF() string {
	for _, p := range []string{
		"/usr/games/stockfish",
		"/usr/bin/stockfish",
		"/usr/local/bin/stockfish",
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if p, err := exec.LookPath("stockfish"); err == nil {
		return p
	}
	return ""
}

func (sf *SF) tx(s string) { fmt.Fprintln(sf.in, s) }

func (sf *SF) expect(tok string) {
	for sf.scan.Scan() {
		if strings.Contains(sf.scan.Text(), tok) {
			return
		}
	}
}


func (sf *SF) bestMove(moves []string, depth int) string {
	if len(moves) == 0 {
		sf.tx("position startpos")
	} else {
		sf.tx("position startpos moves " + strings.Join(moves, " "))
	}
	sf.tx(fmt.Sprintf("go depth %d", depth))
	for sf.scan.Scan() {
		line := sf.scan.Text()
		if strings.HasPrefix(line, "bestmove") {
			parts := strings.Fields(line)
			if len(parts) >= 2 && parts[1] != "(none)" {
				return parts[1]
			}
			return ""
		}
	}
	return ""
}

func (sf *SF) close() {
	sf.tx("quit")
	_ = sf.cmd.Wait()
}


var (
	whiteGlyph = map[chess.PieceType]rune{
		chess.King: '♔', chess.Queen: '♕', chess.Rook: '♖',
		chess.Bishop: '♗', chess.Knight: '♘', chess.Pawn: '♙',
	}
	blackGlyph = map[chess.PieceType]rune{
		chess.King: '♚', chess.Queen: '♛', chess.Rook: '♜',
		chess.Bishop: '♝', chess.Knight: '♞', chess.Pawn: '♟',
	}
)

func pieceGlyph(p chess.Piece) (rune, bool) {
	if p == chess.NoPiece {
		return ' ', false
	}
	if p.Color() == chess.White {
		return whiteGlyph[p.Type()], true
	}
	return blackGlyph[p.Type()], false
}


var (

	bgLight  = tcell.ColorSilver
	bgDark   = tcell.ColorNavy
	bgHl     = tcell.ColorOlive
	bgCheck  = tcell.ColorMaroon
	bgSel    = tcell.ColorGreen
	bgDest   = tcell.ColorTeal
	bgCursor = tcell.ColorPurple


	fgWhite = tcell.ColorYellow
	fgBlack = tcell.ColorAqua


	stLabel  = tcell.StyleDefault.Foreground(tcell.ColorSilver)
	stTitle  = tcell.StyleDefault.Foreground(tcell.ColorTeal).Bold(true)
	stInfo   = tcell.StyleDefault.Foreground(tcell.ColorGreen).Bold(true)
	stWarn   = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	stErr    = tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
	stInput  = tcell.StyleDefault.Foreground(tcell.ColorWhite).Bold(true)
	stHint   = tcell.StyleDefault.Foreground(tcell.ColorNavy)
	stBorder = tcell.StyleDefault.Foreground(tcell.ColorNavy)
	stMove   = tcell.StyleDefault.Foreground(tcell.ColorSilver)
	stWin    = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
)


func put(s tcell.Screen, x, y int, text string, st tcell.Style) {
	for i, ch := range []rune(text) {
		s.SetContent(x+i, y, ch, nil, st)
	}
}

func putC(s tcell.Screen, w, y int, text string, st tcell.Style) {
	runes := []rune(text)
	x := (w - len(runes)) / 2
	if x < 0 {
		x = 0
	}
	for i, ch := range runes {
		s.SetContent(x+i, y, ch, nil, st)
	}
}

func drawBorder(s tcell.Screen, w, h int) {
	for x := 0; x < w; x++ {
		s.SetContent(x, 0, tcell.RuneHLine, nil, stBorder)
		s.SetContent(x, h-1, tcell.RuneHLine, nil, stBorder)
	}
	for y := 0; y < h; y++ {
		s.SetContent(0, y, tcell.RuneVLine, nil, stBorder)
		s.SetContent(w-1, y, tcell.RuneVLine, nil, stBorder)
	}
	s.SetContent(0, 0, tcell.RuneULCorner, nil, stBorder)
	s.SetContent(w-1, 0, tcell.RuneURCorner, nil, stBorder)
	s.SetContent(0, h-1, tcell.RuneLLCorner, nil, stBorder)
	s.SetContent(w-1, h-1, tcell.RuneLRCorner, nil, stBorder)
}



const noSq = chess.Square(64)

func renderBoard(s tcell.Screen, pos *chess.Position, lastMove *chess.Move,
	cursor, selected chess.Square, validDests map[chess.Square]bool) {
	sqMap := pos.Board().SquareMap()


	var checkKing chess.Square = chess.Square(64)
	if lastMove != nil && lastMove.HasTag(chess.Check) {
		for sq, p := range sqMap {
			if p.Type() == chess.King && p.Color() == pos.Turn() {
				checkKing = sq
				break
			}
		}
	}


	for f := 0; f < 8; f++ {
		x := bX + f*cellW + 1
		s.SetContent(x, bY-1, rune('a'+f), nil, stLabel)
		s.SetContent(x, bY+8*cellH, rune('a'+f), nil, stLabel)
	}


	for rank := 7; rank >= 0; rank-- {
		sy := bY + (7-rank)*cellH


		rl := rune('1' + rank)
		s.SetContent(bX-2, sy+cellH/2, rl, nil, stLabel)
		s.SetContent(bX+8*cellW+1, sy+cellH/2, rl, nil, stLabel)

		for file := 0; file < 8; file++ {
			sq := chess.Square(rank*8 + file)


			isLight := (file+rank)%2 != 0
			var bg tcell.Color
			switch {
			case sq == checkKing:
				bg = bgCheck
			case sq == cursor:
				bg = bgCursor
			case sq == selected:
				bg = bgSel
			case validDests[sq]:
				bg = bgDest
			case lastMove != nil && (lastMove.S1() == sq || lastMove.S2() == sq):
				bg = bgHl
			case isLight:
				bg = bgLight
			default:
				bg = bgDark
			}

			base := tcell.StyleDefault.Background(bg)


			for dy := 0; dy < cellH; dy++ {
				for dx := 0; dx < cellW; dx++ {
					s.SetContent(bX+file*cellW+dx, sy+dy, ' ', nil, base)
				}
			}


			p := sqMap[sq]
			if p != chess.NoPiece {
				glyph, isW := pieceGlyph(p)
				fg := fgBlack
				if isW {
					fg = fgWhite
				}
				pst := tcell.StyleDefault.Background(bg).Foreground(fg).Bold(true)
				s.SetContent(bX+file*cellW+1, sy+cellH-1, glyph, nil, pst)
			} else if validDests[sq] {

				dotSt := tcell.StyleDefault.Background(bg).Foreground(tcell.ColorWhite).Bold(true)
				s.SetContent(bX+file*cellW+1, sy+cellH-1, '·', nil, dotSt)
			}
		}
	}
}


const (
	modeHvAI  = 1
	modeAIvAI = 2
)

type GCtx struct {
	g    *chess.Game
	mode int
	diff Diff
	sfW  *SF
	sfB  *SF

	history  []string
	lastMove *chess.Move

	thinking   bool
	moveCh     chan string
	nextMoveAt time.Time


	cursor     chess.Square
	selected   chess.Square
	validDests map[chess.Square]bool

	errMsg string
	status string

	gameOver bool
	paused   bool
}

func newGCtx(mode int, diff Diff, sfW, sfB *SF) *GCtx {
	return &GCtx{
		g:          chess.NewGame(),
		mode:       mode,
		diff:       diff,
		sfW:        sfW,
		sfB:        sfB,
		moveCh:     make(chan string, 1),
		cursor:     chess.Square(1*8 + 4),
		selected:   noSq,
		validDests: map[chess.Square]bool{},
	}
}


func buildValidDests(pos *chess.Position, from chess.Square) map[chess.Square]bool {
	dests := map[chess.Square]bool{}
	for _, m := range pos.ValidMoves() {
		if m.S1() == from {
			dests[m.S2()] = true
		}
	}
	return dests
}


func (gc *GCtx) moveCursor(dFile, dRank int) {
	file := int(gc.cursor)%8 + dFile
	rank := int(gc.cursor)/8 + dRank
	if file < 0 {
		file = 0
	}
	if file > 7 {
		file = 7
	}
	if rank < 0 {
		rank = 0
	}
	if rank > 7 {
		rank = 7
	}
	gc.cursor = chess.Square(rank*8 + file)
}


func (gc *GCtx) confirm() {
	if gc.gameOver || gc.thinking || !gc.isHumanTurn() {
		return
	}
	sq := gc.cursor
	pos := gc.g.Position()
	sqMap := pos.Board().SquareMap()

	if gc.selected != noSq {
		if gc.validDests[sq] {

			from := gc.selected
			gc.selected = noSq
			gc.validDests = map[chess.Square]bool{}
			uci := squareUCI(from) + squareUCI(sq)

			p := sqMap[from]
			toRank := int(sq) / 8
			if p != chess.NoPiece && p.Type() == chess.Pawn && (toRank == 7 || toRank == 0) {
				uci += "q"
			}
			if err := gc.applyUCI(uci); err != nil {
				gc.errMsg = err.Error()
			} else {
				gc.errMsg = ""
				if !gc.gameOver {
					gc.triggerAI()
				}
			}
		} else {

			p := sqMap[sq]
			if p != chess.NoPiece && p.Color() == pos.Turn() {
				gc.selected = sq
				gc.validDests = buildValidDests(pos, sq)
				gc.errMsg = ""
			} else {
				gc.deselect()
			}
		}
	} else {

		p := sqMap[sq]
		if p != chess.NoPiece && p.Color() == pos.Turn() {
			gc.selected = sq
			gc.validDests = buildValidDests(pos, sq)
			gc.errMsg = ""
		}
	}
}


func (gc *GCtx) deselect() {
	gc.selected = noSq
	gc.validDests = map[chess.Square]bool{}
	gc.errMsg = ""
}


func squareUCI(sq chess.Square) string {
	file := int(sq) % 8
	rank := int(sq) / 8
	return string([]byte{byte('a' + file), byte('1' + rank)})
}

func (gc *GCtx) isHumanTurn() bool {
	return gc.mode == modeHvAI && gc.g.Position().Turn() == chess.White
}

func (gc *GCtx) activeSF() *SF {
	if gc.g.Position().Turn() == chess.White {
		return gc.sfW
	}
	return gc.sfB
}

func (gc *GCtx) triggerAI() {
	if gc.thinking {
		return
	}
	gc.thinking = true
	sf := gc.activeSF()
	snap := append([]string{}, gc.history...)
	depth := gc.diff.depth
	go func() {
		gc.moveCh <- sf.bestMove(snap, depth)
	}()
}


func (gc *GCtx) applyUCI(uci string) error {
	uci = strings.TrimSpace(strings.ToLower(uci))
	if len(uci) < 4 {
		return fmt.Errorf("too short — use UCI format: e2e4")
	}


	if len(uci) == 4 {
		fromSq := chess.Square(int(uci[1]-'1')*8 + int(uci[0]-'a'))
		p := gc.g.Position().Board().SquareMap()[fromSq]
		destRank := int(uci[3] - '1')
		if p != chess.NoPiece && p.Type() == chess.Pawn && (destRank == 7 || destRank == 0) {
			uci += "q"
		}
	}

	m, err := chess.UCINotation{}.Decode(gc.g.Position(), uci)
	if err != nil {
		return fmt.Errorf("invalid move: %s", uci)
	}
	if err := gc.g.Move(m); err != nil {
		return fmt.Errorf("illegal move: %s", uci)
	}
	gc.history = append(gc.history, uci)
	gc.lastMove = m


	outcome := gc.g.Outcome()
	if outcome != chess.NoOutcome {
		gc.gameOver = true
		switch outcome {
		case chess.WhiteWon:
			gc.status = "♔  White Wins!"
		case chess.BlackWon:
			gc.status = "♚  Black Wins!"
		case chess.Draw:
			gc.status = "½-½  Draw"
		}
		return nil
	}


	if gc.lastMove != nil && gc.lastMove.HasTag(chess.Check) {
		side := "White"
		if gc.g.Position().Turn() == chess.Black {
			side = "Black"
		}
		gc.status = side + " is in CHECK!"
	} else {
		gc.status = ""
	}
	return nil
}


func (gc *GCtx) reset() {
	gc.g = chess.NewGame()
	gc.history = gc.history[:0]
	gc.lastMove = nil
	gc.errMsg = ""
	gc.status = ""
	gc.gameOver = false
	gc.thinking = false
	gc.selected = noSq
	gc.validDests = map[chess.Square]bool{}
	gc.cursor = chess.Square(1*8 + 4)

	select {
	case <-gc.moveCh:
	default:
	}
}


func renderInfo(s tcell.Screen, gc *GCtx) {
	x, y := iX, bY


	turnStr := "White  ♙"
	turnSt := tcell.StyleDefault.Foreground(fgWhite).Bold(true)
	if gc.g.Position().Turn() == chess.Black {
		turnStr = "Black  ♟"
		turnSt = tcell.StyleDefault.Foreground(fgBlack).Bold(true)
	}
	put(s, x, y, "Turn:", stInfo)
	put(s, x+6, y, turnStr, turnSt)
	y += 2


	if gc.status != "" {
		st := stWarn
		if gc.gameOver {
			st = stWin
		} else if strings.Contains(gc.status, "CHECK") {
			st = stErr
		}
		put(s, x, y, gc.status, st)
	}
	if gc.thinking {
		put(s, x, y+1, "  ⟳ Engine thinking…", stInfo)
	}
	y += 3


	put(s, x, y, "── Move Log ──────────", stInfo)
	y++
	start := 0
	if len(gc.history) > 14 {
		start = len(gc.history) - 14
	}

	if start%2 != 0 {
		start--
	}
	lineNum := 1 + start/2
	for i := start; i < len(gc.history); i += 2 {
		wm := gc.history[i]
		bm := ""
		if i+1 < len(gc.history) {
			bm = gc.history[i+1]
		}
		put(s, x, y, fmt.Sprintf("%2d. %-7s %s", lineNum, wm, bm), stMove)
		y++
		lineNum++
	}


	if gc.mode == modeHvAI && !gc.gameOver {
		inputY := bY + 8*cellH + 1
		if gc.isHumanTurn() {
			curName := squareUCI(gc.cursor)
			if gc.selected == noSq {
				put(s, x, inputY, fmt.Sprintf("Cursor: %s — ↑↓←→ move, Space/Enter select", curName), stInfo)
			} else {
				selName := squareUCI(gc.selected)
				put(s, x, inputY, fmt.Sprintf("From: %s → move to %s  (Esc to cancel)", selName, curName), stWarn)
			}
		} else {
			put(s, x, inputY, "Engine is thinking…", stInfo)
		}
		if gc.errMsg != "" {
			put(s, x, inputY+1, gc.errMsg, stErr)
		}
	}


	if gc.gameOver {
		endY := bY + 8*cellH + 1
		put(s, x, endY, gc.status, stWin)
		put(s, x, endY+1, "R: new game   Q: quit", stHint)
	}
}


func runGame(s tcell.Screen, mode int, diff Diff) {

	var sfW, sfB *SF
	var err error

	if mode == modeHvAI {

		sfB, err = newSF(diff)
		if err != nil {
			s.Fini()
			fmt.Fprintln(os.Stderr, "\nStockfish error:", err)
			os.Exit(1)
		}
	} else {

		sfW, err = newSF(diff)
		if err != nil {
			s.Fini()
			fmt.Fprintln(os.Stderr, "\nStockfish error (white):", err)
			os.Exit(1)
		}
		sfB, err = newSF(diff)
		if err != nil {
			sfW.close()
			s.Fini()
			fmt.Fprintln(os.Stderr, "\nStockfish error (black):", err)
			os.Exit(1)
		}
	}
	defer func() {
		if sfW != nil {
			sfW.close()
		}
		if sfB != nil {
			sfB.close()
		}
	}()

	gc := newGCtx(mode, diff, sfW, sfB)


	if mode == modeAIvAI {
		gc.nextMoveAt = time.Now().Add(600 * time.Millisecond)
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

	ticker := time.NewTicker(time.Second / 60)
	defer ticker.Stop()

	modeStr := "Human (White) vs AI (Black)"
	if mode == modeAIvAI {
		modeStr = "AI vs AI — watch mode"
	}

	for range ticker.C {
		w, h := s.Size()


		var doQuit, doPause, doRestart bool
	drain:
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
					switch {
					case e.Rune() == 'q' || e.Rune() == 'Q':
						doQuit = true

					case e.Key() == tcell.KeyEscape:
						if gc.selected != noSq {
							gc.deselect()
						} else {
							doQuit = true
						}

					case e.Rune() == 'p' || e.Rune() == 'P':
						doPause = true

					case e.Rune() == 'r' || e.Rune() == 'R':
						doRestart = true


					case e.Key() == tcell.KeyUp:
						gc.moveCursor(0, 1)
					case e.Key() == tcell.KeyDown:
						gc.moveCursor(0, -1)
					case e.Key() == tcell.KeyLeft:
						gc.moveCursor(-1, 0)
					case e.Key() == tcell.KeyRight:
						gc.moveCursor(1, 0)


					case e.Key() == tcell.KeyEnter || e.Rune() == ' ':
						gc.confirm()
					}
				}
			default:
				break drain
			}
		}

		if doQuit {
			return
		}
		if doPause {
			gc.paused = !gc.paused
		}
		if doRestart {
			gc.reset()
			if mode == modeAIvAI {
				gc.nextMoveAt = time.Now().Add(600 * time.Millisecond)
			}
		}


		if gc.paused {
			s.Clear()
			drawBorder(s, w, h)
			putC(s, w, h/2-1, "── PAUSED ──", stWarn)
			putC(s, w, h/2, "P to resume  │  R to restart  │  Q to quit", stHint)
			s.Show()
			continue
		}


		select {
		case uci := <-gc.moveCh:
			gc.thinking = false
			if uci != "" && !gc.gameOver {
				if applyErr := gc.applyUCI(uci); applyErr != nil {
					gc.errMsg = "Engine error: " + applyErr.Error()
				} else if mode == modeAIvAI && !gc.gameOver {

					gc.nextMoveAt = time.Now().Add(700 * time.Millisecond)
				}
			}
		default:
		}


		if !gc.gameOver && !gc.paused && !gc.thinking {
			if mode == modeAIvAI {
				if time.Now().After(gc.nextMoveAt) {
					gc.triggerAI()
				}
			} else if mode == modeHvAI && !gc.isHumanTurn() {
				gc.triggerAI()
			}
		}


		s.Clear()
		drawBorder(s, w, h)

		title := fmt.Sprintf(" ♛  CHESS  —  %s  —  %s ", modeStr, diff.label)
		putC(s, w, 0, title, stTitle)

		renderBoard(s, gc.g.Position(), gc.lastMove, gc.cursor, gc.selected, gc.validDests)
		renderInfo(s, gc)


		hint := " ↑↓←→: move cursor  Space/Enter: select/confirm  Esc: cancel  R: restart  Q: quit "
		if mode == modeAIvAI {
			hint = " Watching AI vs AI  │  P: pause  │  R: restart  │  Q: quit "
		}
		putC(s, w, h-1, hint, stHint)

		s.Show()
	}
}


func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println()
	fmt.Println("  ╔═══════════════════════════════════════╗")
	fmt.Println("  ║         CHESS  ♛  Stockfish           ║")
	fmt.Println("  ╚═══════════════════════════════════════╝")
	fmt.Println()


	fmt.Println("  Select mode:")
	fmt.Println("    1  —  Human vs AI  (you play White)")
	fmt.Println("    2  —  AI vs AI     (watch mode)")
	fmt.Println()

	var mode int
	for {
		fmt.Print("  Enter 1 or 2: ")
		line, _ := reader.ReadString('\n')
		if len(line) > 0 {
			switch line[0] {
			case '1':
				mode = modeHvAI
			case '2':
				mode = modeAIvAI
			}
			if mode != 0 {
				break
			}
		}
		fmt.Println("  Please enter 1 or 2.")
	}


	fmt.Println()
	fmt.Println("  Select difficulty:")
	fmt.Println("    1  —  Easy   (depth 3,  skill 2)")
	fmt.Println("    2  —  Medium (depth 8,  skill 10)")
	fmt.Println("    3  —  Hard   (depth 15, skill 20)")
	fmt.Println()

	diffIdx := -1
	for {
		fmt.Print("  Enter 1, 2, or 3: ")
		line, _ := reader.ReadString('\n')
		if len(line) > 0 {
			switch line[0] {
			case '1':
				diffIdx = 0
			case '2':
				diffIdx = 1
			case '3':
				diffIdx = 2
			}
			if diffIdx >= 0 {
				break
			}
		}
		fmt.Println("  Please enter 1, 2, or 3.")
	}
	diff := diffs[diffIdx]


	fmt.Println()
	if mode == modeHvAI {
		fmt.Printf("  Human vs AI — difficulty: %s\n", diff.label)
		fmt.Println()
		fmt.Println("  You play White.  Enter moves in UCI format:")
		fmt.Println("    e2e4  →  pawn to e4")
		fmt.Println("    g1f3  →  knight to f3")
		fmt.Println("    e1g1  →  castle kingside")
		fmt.Println("    e7e8  →  pawn promotes to queen (auto)")
		fmt.Println("    e7e8r →  pawn promotes to rook")
	} else {
		fmt.Printf("  AI vs AI — both engines at %s difficulty.\n", diff.label)
		fmt.Println("  Sit back and watch!")
	}
	fmt.Println()
	fmt.Println("  Controls:  P — pause/resume   R — new game   Q — quit")
	fmt.Printf("\n  Terminal must be at least 80×24.  Press Enter to start…")
	reader.ReadString('\n')


	s, err := tcell.NewScreen()
	if err != nil {
		fmt.Fprintln(os.Stderr, "tcell error:", err)
		os.Exit(1)
	}
	if err := s.Init(); err != nil {
		fmt.Fprintln(os.Stderr, "tcell init error:", err)
		os.Exit(1)
	}
	defer s.Fini()

	s.SetStyle(tcell.StyleDefault)
	s.Clear()

	runGame(s, mode, diff)

	fmt.Println("\n  Thanks for playing!\n")
}
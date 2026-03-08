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
	fps          = 60
	startLives   = 3
	alienCols    = 11
	alienRows    = 5
	alienSpaceX  = 4
	alienSpaceY  = 2
	bulletSpeed  = 0.7
	alienBulletS = 0.35
	playerSpeed  = 0.18
	shieldW      = 4
	numShields   = 4
	shieldRow    = 4
)

var frameDur = time.Second / fps


var (
	stylePlayer    = tcell.StyleDefault.Foreground(tcell.ColorLime).Bold(true)
	styleBullet    = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	styleAlienBlt  = tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
	styleBorder    = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x1a3a5c))
	styleHUD       = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	styleLives     = tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
	styleScore     = tcell.StyleDefault.Foreground(tcell.ColorAqua).Bold(true)
	styleHint      = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x334455))
	styleWin       = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	styleGameOver  = tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
	styleShield    = tcell.StyleDefault.Foreground(tcell.ColorGreen)
	styleShieldDmg = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x446644))
	styleStars     = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x223344))
	styleBoss      = tcell.StyleDefault.Foreground(tcell.ColorFuchsia).Bold(true)
	styleSpark     = tcell.StyleDefault.Foreground(tcell.ColorOrange)
	styleAI        = tcell.StyleDefault.Foreground(tcell.ColorFuchsia).Bold(true)
)

var alienStyles = []tcell.Style{
	tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true),
	tcell.StyleDefault.Foreground(tcell.NewHexColor(0xFF6600)).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorLime).Bold(true),
	tcell.StyleDefault.Foreground(tcell.ColorAqua).Bold(true),
}


var alienShapes = [alienRows][2][2]string{
	{{"▄█▄", "▀█▀"}, {"▄█▄", "▀█▀"}},
	{{"(▀█▀)", "(▄█▄)"}, {"(▀█▀)", "(▄█▄)"}},
	{{"|▓|", "|░|"}, {"|░|", "|▓|"}},
	{{"╔▓╗", "╚▓╝"}, {"╚▓╝", "╔▓╗"}},
	{{"▐█▌", "▌█▐"}, {"▌█▐", "▐█▌"}},
}

var alienScores = [alienRows]int{50, 40, 30, 20, 10}


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


type Star struct {
	x, y  int
	ch    rune
	layer int
}

func generateStars(w, h, n int) []Star {
	stars := make([]Star, n)
	chars := []rune{'.', '·', '∙', '⋅', '*', '✦', '✧', '★'}
	for i := range stars {
		layer := rand.Intn(3)
		stars[i] = Star{
			x:     rand.Intn(w-2) + 1,
			y:     rand.Intn(h-2) + 1,
			ch:    chars[rand.Intn(len(chars))],
			layer: layer,
		}
	}
	return stars
}

func drawStars(s tcell.Screen, stars []Star) {
	for _, st := range stars {
		var style tcell.Style
		switch st.layer {
		case 0:
			style = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x1a2a3a))
		case 1:
			style = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x223344))
		case 2:
			style = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x334466))
		}
		s.SetContent(st.x, st.y, st.ch, nil, style)
	}
}


type Spark struct {
	x, y   float64
	vx, vy float64
	ttl    int
	ch     rune
}

func newExplosion(cx, cy int) []Spark {
	sparks := make([]Spark, 8)
	glyphs := []rune{'*', '✸', '✦', '·', '×', '+', '✺', '⁕'}
	for i := range sparks {
		angle := float64(i) * math.Pi / 4
		spd := 0.3 + rand.Float64()*0.4
		sparks[i] = Spark{
			x: float64(cx), y: float64(cy),
			vx:  math.Cos(angle) * spd,
			vy:  math.Sin(angle) * spd * 0.5,
			ttl: 12 + rand.Intn(10),
			ch:  glyphs[i],
		}
	}
	return sparks
}


type Bullet struct {
	x, y  float64
	vy    float64
	alive bool
}


type ShieldBlock struct {
	x, y  int
	hp    int
	alive bool
}

func buildShields(w, h int) []ShieldBlock {
	playerRow := h - 3
	shieldRowY := playerRow - shieldRow
	totalW := numShields*shieldW + (numShields-1)*3
	startX := (w - totalW) / 2
	var blocks []ShieldBlock
	for i := 0; i < numShields; i++ {
		bx := startX + i*(shieldW+3)
		for dx := 0; dx < shieldW; dx++ {
			for dy := 0; dy < 2; dy++ {
				blocks = append(blocks, ShieldBlock{
					x: bx + dx, y: shieldRowY + dy,
					hp: 3, alive: true,
				})
			}
		}
	}
	return blocks
}

func drawShields(s tcell.Screen, shields []ShieldBlock) {
	shieldChars := [4]rune{'█', '▓', '▒', '░'}
	for _, b := range shields {
		if !b.alive {
			continue
		}
		idx := 3 - b.hp
		if idx < 0 {
			idx = 0
		}
		if idx > 3 {
			idx = 3
		}
		style := styleShield
		if b.hp < 3 {
			style = styleShieldDmg
		}
		s.SetContent(b.x, b.y, shieldChars[idx], nil, style)
	}
}


type Boss struct {
	x, y    float64
	vx      float64
	hp      int
	alive   bool
	frame   int
	scoreTk int
}

func newBoss(w int) *Boss {
	return &Boss{x: 2, y: 2, vx: 0.4, hp: 5, alive: true, scoreTk: 500}
}

func (b *Boss) update(w int) {
	b.x += b.vx
	b.frame++
	if b.x <= 1 || b.x >= float64(w)-12 {
		b.vx = -b.vx
	}
}

func (b *Boss) draw(s tcell.Screen) {
	if !b.alive {
		return
	}
	f := (b.frame / 4) % 2
	var shape string
	if f == 0 {
		shape = "╔═[◉◉]═╗"
	} else {
		shape = "║═[◉◉]═║"
	}
	drawStr(s, int(b.x), int(b.y), shape, styleBoss)

	hp := ""
	for i := 0; i < b.hp; i++ {
		hp += "♥"
	}
	drawStr(s, int(b.x)+1, int(b.y)+1, hp, styleLives)
}


type Alien struct {
	alive bool
	row   int
	col   int
}

type AlienGrid struct {
	aliens   [alienRows][alienCols]Alien
	ox, oy   float64
	vx       float64
	frame    int
	moveStep float64
	stepX    float64
	dropDown bool
	dropY    float64
	cellW    int
	cellH    int
	startX   int
}

func newAlienGrid(w, h, wave int) *AlienGrid {
	g := &AlienGrid{}
	g.cellW = alienSpaceX + 3
	g.cellH = alienSpaceY + 1
	totalW := alienCols*g.cellW - 1
	g.startX = (w - totalW) / 2
	g.ox = float64(g.startX)
	g.oy = 2.0
	spd := 0.008 + float64(wave)*0.004
	if spd > 0.06 {
		spd = 0.06
	}
	g.vx = spd
	g.stepX = spd
	for r := 0; r < alienRows; r++ {
		for c := 0; c < alienCols; c++ {
			g.aliens[r][c] = Alien{alive: true, row: r, col: c}
		}
	}
	return g
}

func (g *AlienGrid) remaining() int {
	n := 0
	for r := 0; r < alienRows; r++ {
		for c := 0; c < alienCols; c++ {
			if g.aliens[r][c].alive {
				n++
			}
		}
	}
	return n
}


func (g *AlienGrid) updateSpeed(total, alive int) {
	if total <= 0 {
		return
	}
	frac := float64(alive) / float64(total)

	boost := 1.0 + (1.0-frac)*3.0
	g.stepX = g.vx * boost
}

func (g *AlienGrid) leftmost() int {
	for c := 0; c < alienCols; c++ {
		for r := 0; r < alienRows; r++ {
			if g.aliens[r][c].alive {
				return c
			}
		}
	}
	return alienCols
}

func (g *AlienGrid) rightmost() int {
	for c := alienCols - 1; c >= 0; c-- {
		for r := 0; r < alienRows; r++ {
			if g.aliens[r][c].alive {
				return c
			}
		}
	}
	return -1
}

func (g *AlienGrid) update(w int) {
	g.frame++
	if g.dropDown {
		g.oy += 0.5
		g.dropY -= 0.5
		if g.dropY <= 0 {
			g.dropDown = false
			g.stepX = -g.stepX
		}
		return
	}

	g.ox += g.stepX
	lc := g.leftmost()
	rc := g.rightmost()
	lx := int(g.ox) + lc*g.cellW
	rx := int(g.ox) + rc*g.cellW + 2
	if g.stepX > 0 && rx >= w-2 {
		g.dropDown = true
		g.dropY = 1.5
	} else if g.stepX < 0 && lx <= 1 {
		g.dropDown = true
		g.dropY = 1.5
	}
}

func (g *AlienGrid) draw(s tcell.Screen) {
	animFrame := (g.frame / 20) % 2
	for r := 0; r < alienRows; r++ {
		for c := 0; c < alienCols; c++ {
			if !g.aliens[r][c].alive {
				continue
			}
			x := int(g.ox) + c*g.cellW
			y := int(g.oy) + r*g.cellH
			shape := alienShapes[r][animFrame][0]
			st := alienStyles[r]
			drawStr(s, x, y, shape, st)
		}
	}
}


func (g *AlienGrid) hitTest(bx, by int) (int, int) {
	for r := 0; r < alienRows; r++ {
		for c := 0; c < alienCols; c++ {
			if !g.aliens[r][c].alive {
				continue
			}
			ax := int(g.ox) + c*g.cellW
			ay := int(g.oy) + r*g.cellH
			if bx >= ax && bx <= ax+2 && by == ay {
				return r, c
			}
		}
	}
	return -1, -1
}


func (g *AlienGrid) bottomY() int {
	for r := alienRows - 1; r >= 0; r-- {
		for c := 0; c < alienCols; c++ {
			if g.aliens[r][c].alive {
				return int(g.oy) + r*g.cellH
			}
		}
	}
	return 0
}


func (g *AlienGrid) randomShooter() (int, int, bool) {
	type slot struct{ r, c int }
	var candidates []slot
	for c := 0; c < alienCols; c++ {
		for r := alienRows - 1; r >= 0; r-- {
			if g.aliens[r][c].alive {
				candidates = append(candidates, slot{r, c})
				break
			}
		}
	}
	if len(candidates) == 0 {
		return 0, 0, false
	}
	pick := candidates[rand.Intn(len(candidates))]
	x := int(g.ox) + pick.c*g.cellW + 1
	y := int(g.oy) + pick.r*g.cellH + 1
	return x, y, true
}


type Player struct {
	x, y   float64
	lives  int
	score  int
	width  int
	xLeft  int
	xRight int
}

func newPlayer(w, h int) *Player {
	return &Player{
		x: float64(w/2 - 2), y: float64(h - 3),
		lives: startLives, width: 5,
		xLeft: 1, xRight: w - 2,
	}
}

func (p *Player) draw(s tcell.Screen) {
	x := int(p.x)
	y := int(p.y)

	drawStr(s, x, y, "  ▲  ", stylePlayer)
	drawStr(s, x, y+1, "▐███▌", stylePlayer)
}

func (p *Player) center() int {
	return int(p.x) + p.width/2
}


func drawHUD(s tcell.Screen, score, lives, wave, w int, aiMode bool) {
	scoreStr := fmt.Sprintf(" ★ %07d", score)
	drawStr(s, 1, 0, scoreStr, styleScore)
	waveStr := fmt.Sprintf(" WAVE %d ", wave)
	if aiMode {
		waveStr = fmt.Sprintf(" WAVE %d  🤖 AI ", wave)
	}
	drawCentered(s, w, 0, waveStr, styleHUD)
	livesStr := " "
	for i := 0; i < lives; i++ {
		livesStr += "♥ "
	}
	runes := []rune(livesStr)
	drawStr(s, w-len(runes)-1, 0, livesStr, styleLives)
}


type Game struct {
	s        tcell.Screen
	w, h     int
	player   *Player
	grid     *AlienGrid
	boss     *Boss
	shields  []ShieldBlock
	bullets  []*Bullet
	aBullets []*Bullet
	sparks   []Spark
	stars    []Star
	wave     int
	total    int
	paused   bool
	over     bool
	victory  bool


	alienFireTimer float64
	alienFireRate  float64
	bossTimer      float64


	aiMode bool
	ai     AIBrain
}





type AIBrain struct {
	targetX      float64
	reactionTick int
	aimJitter    float64
	dodging      bool
}


type AISignals struct {
	moveLeft  bool
	moveRight bool
	fire      bool
}

func (ai *AIBrain) think(g *Game, now time.Time) AISignals {
	px := g.player.x + float64(g.player.width)/2
	py := g.player.y



	bestThreatDist := math.MaxFloat64
	threatBulletX := -1.0
	for _, b := range g.aBullets {
		if !b.alive {
			continue
		}

		distY := py - b.y
		if distY <= 0 {
			continue
		}

		bx := b.x
		if bx >= g.player.x-1 && bx <= g.player.x+float64(g.player.width)+1 {
			if distY < bestThreatDist {
				bestThreatDist = distY
				threatBulletX = bx
			}
		}
	}

	ai.dodging = threatBulletX >= 0 && bestThreatDist < 14


	ai.reactionTick--
	if ai.reactionTick <= 0 {
		ai.reactionTick = 4 + rand.Intn(4)
		ai.aimJitter = (rand.Float64() - 0.5) * 2.5

		if ai.dodging {

			if threatBulletX < px {
				ai.targetX = px + 8
			} else {
				ai.targetX = px - 8
			}
		} else if g.boss != nil && g.boss.alive {

			ai.targetX = g.boss.x + 4 + ai.aimJitter
		} else {

			best := -1
			bestScore := -1
			for c := 0; c < alienCols; c++ {
				for r := 0; r < alienRows; r++ {
					if g.grid.aliens[r][c].alive {
						sc := alienScores[r]
						ax := int(g.grid.ox) + c*g.grid.cellW + 1

						dist := math.Abs(float64(ax) - px)
						weight := sc*10 - int(dist)
						if weight > bestScore {
							bestScore = weight
							best = c
						}
						break
					}
				}
			}
			if best >= 0 {
				ax := float64(int(g.grid.ox)+best*g.grid.cellW) + 1 + ai.aimJitter
				ai.targetX = ax
			}
		}

		ai.targetX = clamp(ai.targetX, float64(g.player.xLeft+1), float64(g.player.xRight-g.player.width-1))
	}


	var sig AISignals
	diff := ai.targetX - px
	if diff < -0.8 {
		sig.moveLeft = true
	} else if diff > 0.8 {
		sig.moveRight = true
	}

	if !ai.dodging && math.Abs(diff) < 3.0 {
		sig.fire = true
	}
	return sig
}

func newGame(s tcell.Screen, wave int) *Game {
	w, h := s.Size()
	g := &Game{
		s:    s,
		w:    w,
		h:    h,
		wave: wave,
	}
	g.player = newPlayer(w, h)
	g.grid = newAlienGrid(w, h, wave)
	g.total = alienRows * alienCols
	g.shields = buildShields(w, h)
	g.stars = generateStars(w, h, 60)
	g.alienFireRate = math.Max(0.8, 2.5-float64(wave)*0.3)
	g.alienFireTimer = g.alienFireRate
	g.bossTimer = 20.0 + rand.Float64()*15.0
	return g
}

func (g *Game) spawnBullet(x, y int) {

	for _, b := range g.bullets {
		if b.alive {
			return
		}
	}
	g.bullets = append(g.bullets, &Bullet{x: float64(x), y: float64(y), vy: -bulletSpeed, alive: true})
}

func (g *Game) updateBullets(dt float64) {
	for _, b := range g.bullets {
		if !b.alive {
			continue
		}
		b.y += b.vy * 60 * dt
		bx, by := int(math.Round(b.x)), int(math.Round(b.y))
		if by <= 1 {
			b.alive = false
			continue
		}

		if g.boss != nil && g.boss.alive {
			bossX := int(g.boss.x)
			bossY := int(g.boss.y)
			if bx >= bossX && bx <= bossX+8 && by >= bossY && by <= bossY+1 {
				b.alive = false
				g.boss.hp--
				g.sparks = append(g.sparks, newExplosion(bx, by)...)
				if g.boss.hp <= 0 {
					g.boss.alive = false
					g.player.score += g.boss.scoreTk
					g.sparks = append(g.sparks, newExplosion(bossX+4, bossY)...)
				}
				continue
			}
		}

		r, c := g.grid.hitTest(bx, by)
		if r >= 0 {
			g.grid.aliens[r][c].alive = false
			b.alive = false
			g.player.score += alienScores[r]
			g.sparks = append(g.sparks, newExplosion(bx, by)...)
			continue
		}

		for i := range g.shields {
			if g.shields[i].alive && g.shields[i].x == bx && g.shields[i].y == by {
				b.alive = false
				g.shields[i].hp--
				if g.shields[i].hp <= 0 {
					g.shields[i].alive = false
				}
				break
			}
		}
	}

	live := g.bullets[:0]
	for _, b := range g.bullets {
		if b.alive {
			live = append(live, b)
		}
	}
	g.bullets = live
}

func (g *Game) updateAlienBullets(dt float64) {

	g.alienFireTimer -= dt
	if g.alienFireTimer <= 0 {
		g.alienFireTimer = g.alienFireRate
		ax, ay, ok := g.grid.randomShooter()
		if ok {
			g.aBullets = append(g.aBullets, &Bullet{x: float64(ax), y: float64(ay), vy: alienBulletS, alive: true})
		}
	}

	for _, b := range g.aBullets {
		if !b.alive {
			continue
		}
		b.y += b.vy * 60 * dt
		bx, by := int(math.Round(b.x)), int(math.Round(b.y))
		if by >= g.h-1 {
			b.alive = false
			continue
		}

		px := int(g.player.x)
		py := int(g.player.y)
		if bx >= px && bx <= px+4 && (by == py || by == py+1) {
			b.alive = false
			g.player.lives--
			g.sparks = append(g.sparks, newExplosion(bx, by)...)
			if g.player.lives <= 0 {
				g.over = true
			}
			continue
		}

		for i := range g.shields {
			if g.shields[i].alive && g.shields[i].x == bx && g.shields[i].y == by {
				b.alive = false
				g.shields[i].hp--
				if g.shields[i].hp <= 0 {
					g.shields[i].alive = false
				}
				break
			}
		}
	}
	live := g.aBullets[:0]
	for _, b := range g.aBullets {
		if b.alive {
			live = append(live, b)
		}
	}
	g.aBullets = live
}

func (g *Game) updateSparks() {
	live := g.sparks[:0]
	for i := range g.sparks {
		sp := &g.sparks[i]
		sp.x += sp.vx
		sp.y += sp.vy
		sp.ttl--
		if sp.ttl > 0 {
			live = append(live, *sp)
		}
	}
	g.sparks = live
}

func (g *Game) drawSparks() {
	for _, sp := range g.sparks {
		x, y := int(sp.x), int(sp.y)
		if x > 0 && x < g.w-1 && y > 0 && y < g.h-1 {
			alpha := float64(sp.ttl) / 20.0
			var st tcell.Style
			if alpha > 0.6 {
				st = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
			} else if alpha > 0.3 {
				st = styleSpark
			} else {
				st = tcell.StyleDefault.Foreground(tcell.NewHexColor(0x663300))
			}
			g.s.SetContent(x, y, sp.ch, nil, st)
		}
	}
}

func (g *Game) drawBullets() {
	for _, b := range g.bullets {
		if b.alive {
			g.s.SetContent(int(b.x), int(b.y), '|', nil, styleBullet)
		}
	}
	for _, b := range g.aBullets {
		if b.alive {
			g.s.SetContent(int(b.x), int(b.y), '¦', nil, styleAlienBlt)
		}
	}
}


func runGame(s tcell.Screen, aiMode bool) (playAgain bool, nextAIMode bool) {
	w, h := s.Size()
	g := newGame(s, 1)
	g.w, g.h = w, h
	g.aiMode = aiMode

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



	var lastLeft, lastRight, lastFire time.Time
	keyHoldWindow := 120 * time.Millisecond
	lastFireTime := time.Now().Add(-time.Second)
	fireCooldown := 500 * time.Millisecond

	ticker := time.NewTicker(frameDur)
	defer ticker.Stop()
	lastTick := time.Now()

	for range ticker.C {
		now := time.Now()
		dt := now.Sub(lastTick).Seconds()
		lastTick = now

		w, h = s.Size()
		g.w, g.h = w, h

	drainLoop:
		for {
			select {
			case ev, ok := <-evCh:
				if !ok {
					return false, g.aiMode
				}
				switch e := ev.(type) {
				case *tcell.EventResize:
					s.Sync()
					w, h = s.Size()
					g.w, g.h = w, h
					g.shields = buildShields(w, h)
					g.stars = generateStars(w, h, 60)
					g.player.xRight = w - 2
					g.player.y = float64(h - 3)
				case *tcell.EventKey:
					switch {
					case e.Key() == tcell.KeyEscape || e.Rune() == 'q' || e.Rune() == 'Q':
						return false, g.aiMode
					case e.Rune() == 'p' || e.Rune() == 'P':
						g.paused = !g.paused
					case e.Rune() == 't' || e.Rune() == 'T':
						g.aiMode = !g.aiMode
					case e.Key() == tcell.KeyLeft || e.Rune() == 'a' || e.Rune() == 'A':
						lastLeft = time.Now()
					case e.Key() == tcell.KeyRight || e.Rune() == 'd' || e.Rune() == 'D':
						lastRight = time.Now()
					case e.Rune() == ' ':
						lastFire = time.Now()
					case e.Key() == tcell.KeyEnter:
						lastFire = time.Now()
					}
				}
			default:
				break drainLoop
			}
		}
		if g.paused {
			s.Clear()
			drawBorder(s, w, h)
			drawStars(s, g.stars)
			lines := []string{"── PAUSED ──", "P to resume  •  Q to quit"}
			sy := h/2 - 1
			for i, ln := range lines {
				drawCentered(s, w, sy+i, ln, styleHUD)
			}
			s.Show()
			continue
		}

		if g.over {

			for i := 0; i < 8; i++ {
				s.Clear()
				drawBorder(s, w, h)
				drawCentered(s, w, h/2-2, "╔══════════════════╗", styleGameOver)
				drawCentered(s, w, h/2-1, "║   GAME  OVER     ║", styleGameOver)
				drawCentered(s, w, h/2, "║                  ║", styleGameOver)
				drawCentered(s, w, h/2+1, fmt.Sprintf("║  SCORE: %07d  ║", g.player.score), styleGameOver)
				drawCentered(s, w, h/2+2, "╚══════════════════╝", styleGameOver)
				drawCentered(s, w, h/2+4, "  R to restart  •  Q to quit  ", styleHint)
				s.Show()
				time.Sleep(80 * time.Millisecond)
				s.Clear()
				drawBorder(s, w, h)
				s.Show()
				time.Sleep(60 * time.Millisecond)
			}

			for {
				ev := s.PollEvent()
				if ev == nil {
					return false, g.aiMode
				}
				if e, ok := ev.(*tcell.EventKey); ok {
					switch {
					case e.Rune() == 'r' || e.Rune() == 'R':
						return true, g.aiMode
					case e.Key() == tcell.KeyEscape || e.Rune() == 'q' || e.Rune() == 'Q':
						return false, g.aiMode
					}
				}
			}
		}

		if g.victory {

			for i := 0; i < 12; i++ {
				s.Clear()
				drawBorder(s, w, h)
				drawCentered(s, w, h/2-1, fmt.Sprintf("  ✦ WAVE %d CLEARED ✦  ", g.wave), styleWin)
				drawCentered(s, w, h/2+1, "  Preparing next wave...  ", styleHUD)
				s.Show()
				time.Sleep(70 * time.Millisecond)
				s.Clear()
				drawBorder(s, w, h)
				s.Show()
				time.Sleep(50 * time.Millisecond)
			}

			nextWave := g.wave + 1
			score := g.player.score + 1000
			lives := g.player.lives
			ai := g.aiMode
			g = newGame(s, nextWave)
			g.player.score = score
			g.player.lives = lives
			g.aiMode = ai
			continue
		}


		speed := playerSpeed * 60.0 * dt

		if g.aiMode {

			sig := g.ai.think(g, now)
			if sig.moveLeft {
				g.player.x = clamp(g.player.x-speed, float64(g.player.xLeft), float64(g.player.xRight-g.player.width))
			}
			if sig.moveRight {
				g.player.x = clamp(g.player.x+speed, float64(g.player.xLeft), float64(g.player.xRight-g.player.width))
			}
			if sig.fire && time.Since(lastFireTime) >= fireCooldown {
				g.spawnBullet(g.player.center(), int(g.player.y))
				lastFireTime = now
			}
		} else {

			if now.Sub(lastLeft) < keyHoldWindow {
				g.player.x = clamp(g.player.x-speed, float64(g.player.xLeft), float64(g.player.xRight-g.player.width))
			}
			if now.Sub(lastRight) < keyHoldWindow {
				g.player.x = clamp(g.player.x+speed, float64(g.player.xLeft), float64(g.player.xRight-g.player.width))
			}
			if now.Sub(lastFire) < keyHoldWindow && time.Since(lastFireTime) >= fireCooldown {
				g.spawnBullet(g.player.center(), int(g.player.y))
				lastFireTime = now
			}
		}


		g.bossTimer -= dt
		if g.bossTimer <= 0 && (g.boss == nil || !g.boss.alive) {
			g.boss = newBoss(w)
			g.bossTimer = 25.0 + rand.Float64()*20.0
		}
		if g.boss != nil && g.boss.alive {
			g.boss.update(w)
		}

		g.grid.update(w)
		g.grid.updateSpeed(g.total, g.grid.remaining())
		g.updateBullets(dt)
		g.updateAlienBullets(dt)
		g.updateSparks()


		if g.grid.bottomY() >= int(g.player.y)-1 {
			g.player.lives = 0
			g.over = true
		}


		if g.grid.remaining() == 0 {
			g.victory = true
		}


		s.Clear()
		drawStars(s, g.stars)
		drawBorder(s, w, h)
		drawHUD(s, g.player.score, g.player.lives, g.wave, w, g.aiMode)

		if g.boss != nil && g.boss.alive {
			g.boss.draw(s)
		}
		g.grid.draw(s)
		drawShields(s, g.shields)
		g.player.draw(s)
		g.drawBullets()
		g.drawSparks()


		if g.aiMode {
			tx := int(g.ai.targetX)
			if tx > 0 && tx < w-1 {
				s.SetContent(tx, int(g.player.y)-1, '▼', nil, styleAI)
			}
			hint := " T:toggle-AI  P:pause  Q:quit "
			drawCentered(s, w, h-1, hint, styleAI)
		} else {
			drawCentered(s, w, h-1, " A/←:left  D/→:right  SPC:fire  T:AI-mode  P:pause  Q:quit ", styleHint)
		}
		s.Show()
	}
	return false, g.aiMode
}


func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println()
	fmt.Println("  ╔══════════════════════════════════╗")
	fmt.Println("  ║   👾  ALIEN SHOOTER (terminal)   ║")
	fmt.Println("  ╚══════════════════════════════════╝")
	fmt.Println()
	fmt.Println("  Controls:")
	fmt.Println("    A / ←      — move left")
	fmt.Println("    D / →      — move right")
	fmt.Println("    SPACE      — fire")
	fmt.Println("    T          — toggle AI mode mid-game")
	fmt.Println("    P          — pause / resume")
	fmt.Println("    Q / Esc    — quit")
	fmt.Println()
	fmt.Println("  Scoring:")
	fmt.Println("    Top row    —  50 pts    Middle  —  30 pts")
	fmt.Println("    Bottom row —  10 pts    Boss    — 500 pts")
	fmt.Println("    Wave clear — 1000 pts bonus")
	fmt.Println()
	fmt.Println("  ┌─────────────────────────────────┐")
	fmt.Println("  │  Mode:  [1] Player   [2] Watch AI│")
	fmt.Println("  └─────────────────────────────────┘")
	fmt.Print("  Enter 1 or 2: ")
	choice, _ := reader.ReadString('\n')
	aiMode := len(choice) > 0 && (choice[0] == '2')

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
	s.HideCursor()
	s.Clear()

	for {
		again, nextAI := runGame(s, aiMode)
		if !again {
			break
		}
		aiMode = nextAI
	}

	s.Fini()
	fmt.Println("\n  Thanks for playing!\n")
}
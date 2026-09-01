# frozen_string_literal: true

# Prose claims, checked against the source and the results they describe.
#
# scripts/check_numbers.py looks for quoted figures in the README and stops
# there. It says so: it does not check claims written in words, and it does not
# check a number in the prose against the constant in the code it is describing.
# Four claims here are of that kind, and all four would go stale silently.
#
#   the drift multiples    open-loop.csv, k=40 against k=1
#   the random baseline    the README against bench/figures.py
#   the timestep           the README against wm/envs.py
#   the episode count      the README against experiments/main.py
#
# Run: ruby verify/degradation.rb .

root = ARGV[0] || "."
failures = 0

# Ruby 2.6 opens files as US-ASCII, and this README has minus signs that are not
# hyphens in it, so the encoding is not optional. Emphasis markers come out
# because a bolded number should not read as a missing one.
def read(path)
  File.read(path, encoding: "UTF-8").gsub("**", "").gsub(/\s+/, " ")
end

def csv(path)
  lines = File.read(path, encoding: "UTF-8").strip.split(/\r?\n/)
  header = lines.shift.split(",")
  lines.map { |l| header.zip(l.split(",")).to_h }
end

def median(xs)
  s = xs.sort
  m = s.length / 2
  s.length.odd? ? s[m] : (s[m - 1] + s[m]) / 2.0
end

readme = read(File.join(root, "README.md"))
ol = csv(File.join(root, "results", "open-loop.csv"))

def mae(rows, mode, step)
  rows.select { |r| r["mode"] == mode && r["step"].to_i == step }
      .map { |r| r["reward_mae"].to_f }
end

RECON = "recon (Dreamer style)"
NORECON = "no-recon (MuZero style)"

puts "drift from k=1 to k=40, medians over 3 seeds"
claim = readme[/degraded to ([\d.]+) times its one step error by then and no-recon to ([\d.]+) times/, 0]
claimed = readme.scan(/degraded to ([\d.]+) times its one step error by then and no-recon to ([\d.]+) times/).flatten
if claimed.length != 2
  puts "  FAIL: README no longer makes the drift claim in that form"
  failures += 1
else
  [[RECON, claimed[0]], [NORECON, claimed[1]]].each do |mode, said|
    one = median(mae(ol, mode, 1))
    forty = median(mae(ol, mode, 40))
    got = forty / one
    ok = format("%.1f", got) == said
    failures += 1 unless ok
    puts format("  %-24s %.4f -> %.4f  %.2f times, README says %s  %s",
                mode, one, forty, got, said, ok ? "ok" : "FAIL")
  end
end
puts "  claim read: #{claim}" if claim

# The baseline the figure draws and the baseline the prose quotes have to be the
# same number. verify/pendulum measures whether that number is right; this is
# only about the two places agreeing.
puts "\nthe random policy baseline, README against bench/figures.py"
src = File.read(File.join(root, "bench", "figures.py"), encoding: "UTF-8")
drawn = src[/^RANDOM_RETURN\s*=\s*(-?[\d.]+)/, 1]
# The README writes a minus sign, not a hyphen.
said = readme[/random policy scores about .([\d.]+)/, 1]
if drawn.nil? || said.nil?
  puts "  FAIL: could not find the baseline in #{drawn.nil? ? 'bench/figures.py' : 'README.md'}"
  failures += 1
else
  ok = drawn.to_f.abs.round(4) == said.to_f.round(4)
  failures += 1 unless ok
  puts format("  bench/figures.py %s, README %s  %s", drawn, said, ok ? "ok" : "FAIL")
end

# The argument about the imagination horizon is arithmetic on the timestep, so
# it breaks if the timestep in wm/envs.py ever changes.
puts "\nthe timestep, README against wm/envs.py"
dt = File.read(File.join(root, "wm", "envs.py"), encoding: "UTF-8")[/^DT\s*=\s*([\d.]+)/, 1].to_f
said_dt = readme[/At dt=([\d.]+)/, 1]
said_secs = readme[/that is ([\d.]+) seconds/, 1]
ok = said_dt && said_dt.to_f == dt
failures += 1 unless ok
puts format("  wm/envs.py DT %.2f, README dt=%s  %s", dt, said_dt || "absent", ok ? "ok" : "FAIL")
ok = said_secs && (said_secs.to_f - 15 * dt).abs < 1e-9
failures += 1 unless ok
puts format("  15 steps is %.2f seconds, README says %s  %s",
            15 * dt, said_secs || "absent", ok ? "ok" : "FAIL")

# The animation is described as the median of 128 episodes. 128 is an argument
# in experiments/main.py, not a round number someone liked.
puts "\nthe episode count behind the open loop numbers"
main = File.read(File.join(root, "experiments", "main.py"), encoding: "UTF-8")
n = main[/open_loop_error\(rssm,\s*(\d+)/, 1]
said_n = readme[/the median of the (\d+) the/, 1]
ok = n && said_n && n == said_n
failures += 1 unless ok
puts format("  experiments/main.py evaluates %s episodes, README says %s  %s",
            n || "absent", said_n || "absent", ok ? "ok" : "FAIL")

if failures.positive?
  puts "\n#{failures} checks failed"
  exit 1
end
puts "\nRuby confirms the prose still describes the code and the results it is about"

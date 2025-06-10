# frozen_string_literal: true

RSpec.describe 'shrinking' do
  include Hypothesis::Debug
  include Hypothesis::Possibilities

  it 'finds a small list' do
    ls, = find { any(arrays(of: integers)).length >= 2 }
    puts "DEBUG: Found list: #{ls.inspect}"
    puts "DEBUG: List length: #{ls.length}"
    
    # Debug: let's see what the draws look like when we replay this
    if ls.length > 2
      puts "DEBUG: Failure case - array too long!"
      puts "DEBUG: Let's examine draw structure..."
      # We can't easily access the draws from here, but we know something went wrong
    end
    
    expect(ls).to eq([0, 0])
  end

  it 'shrinks a list to its last element' do
    10.times do
      @original_target = nil

      ls, = find do
        v = any(arrays(of: integers))

        if v.length >= 5 && @original_target.nil? && v[-1] > 0
          @original_target = v
        end
        !@original_target.nil? && v && v[-1] == @original_target[-1]
      end

      puts "DEBUG: Shrunk list: #{ls.inspect}"
      puts "DEBUG: Shrunk list length: #{ls.length}"
      puts "DEBUG: Original target: #{@original_target.inspect}"
      expect(ls.length).to eq(1)
    end
  end
end

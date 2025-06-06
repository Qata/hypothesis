# frozen_string_literal: true

RSpec.describe 'float possibles' do
  they 'can generate basic floats' do
    hypothesis do
      f = any(floats)
      expect(f).to be_a(Float)
    end
  end

  they 'respect upper bounds' do
    hypothesis do
      expect(any(floats(max: 100.0))).to be <= 100.0
    end
  end

  they 'respect lower bounds' do
    hypothesis do
      expect(any(floats(min: -100.0))).to be >= -100.0
    end
  end

  they 'respect both bounds at once' do
    hypothesis do
      f = any floats(min: 0.0, max: 100.0)
      expect(f).to be <= 100.0
      expect(f).to be >= 0.0
    end
  end

  they 'can generate finite values when NaN is not allowed' do
    hypothesis do
      f = any floats(allow_nan: false)
      expect(f).to be_finite unless f.infinite?
    end
  end

  they 'can generate finite values when infinity is not allowed' do
    hypothesis do
      f = any floats(allow_infinity: false)
      expect(f).to be_finite
    end
  end

  they 'generate small positive numbers' do
    hypothesis do
      f = any floats(min: 0.0, max: 1.0)
      expect(f).to be >= 0.0
      expect(f).to be <= 1.0
    end
  end

  they 'generate negative numbers' do
    hypothesis do
      f = any floats(min: -1.0, max: 0.0)
      expect(f).to be >= -1.0
      expect(f).to be <= 0.0
    end
  end

  they 'can generate very small numbers' do
    hypothesis do
      f = any floats(min: 0.0, max: 1e-100)
      expect(f).to be >= 0.0
      expect(f).to be <= 1e-100
    end
  end

  they 'can generate very large numbers' do
    hypothesis do
      f = any floats(min: 1e100, max: 1e200)
      expect(f).to be >= 1e100
      expect(f).to be <= 1e200
    end
  end
end
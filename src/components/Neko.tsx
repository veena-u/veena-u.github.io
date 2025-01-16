import React, { useState, useEffect, useRef } from 'react';
import type { FC } from 'react';
interface Position {
  x: number;
  y: number;
}

const NekoComponent: FC = () => {
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const [mousePos, setMousePos] = useState<Position>({ x: 0, y: 0 });
  const [sprite, setSprite] = useState<string>('awake');
  const [frame, setFrame] = useState<number>(1);
  const [isWaiting, setIsWaiting] = useState<boolean>(false);
  const [mounted, setMounted] = useState<boolean>(false);
  const [idleState, setIdleState] = useState<number>(0);
  const [stateCount, setStateCount] = useState<number>(0);
  const animationFrameRef = useRef<number | undefined>(undefined);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const isMovingRef = useRef<boolean>(false);

  // Constants
  const SPEED = 2;
  const FRAME_INTERVAL = 200;
  const CAT_SIZE = 40;
  const STATE_DURATION = 30; 

  const IDLE_STATES = [
    { maxState: 2, sprite: 'awake' },
    { maxState: 5, sprite: 'scratch' },
    { maxState: 8, sprite: 'wash' },
    { maxState: 11, sprite: 'yawn' },
    { maxState: Infinity, sprite: 'sleep' }
  ];

  const getSpriteName = (baseName: string, frame: number): string => {
    return baseName === 'awake' ? 'awake' : `${baseName}${frame}`;
  };

  const handleIdle = () => {
    setStateCount(prev => prev + 1);

    if (stateCount >= STATE_DURATION) {
      setStateCount(0);
      setIdleState(prev => prev + 1);
    }

    // Find the appropriate sprite for current idle state
    const currentState = IDLE_STATES.find(state => idleState <= state.maxState);
    setSprite(currentState?.sprite || 'sleep');
  };

  useEffect(() => {
    setMounted(true);
    if (typeof window !== 'undefined') {
      setPosition({
        x: window.innerWidth / 2,
        y: window.innerHeight / 2
      });
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const handleMouseMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [mounted]);

  useEffect(() => {
    if (!mounted) return;

    const handleClick = () => {
      setIsWaiting(!isWaiting);
      setIdleState(0);
      setStateCount(0);
    };

    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [isWaiting, mounted]);

  useEffect(() => {
    if (!mounted) return;

    const animate = (currentTime: number) => {
      if (!lastTimeRef.current) lastTimeRef.current = currentTime;
      const deltaTime = currentTime - lastTimeRef.current;

      if (deltaTime > FRAME_INTERVAL) {
        frameCountRef.current = (frameCountRef.current + 1) % 2;
        setFrame(frameCountRef.current + 1);
        lastTimeRef.current = currentTime;
      }

      if (!isWaiting) {
        const dx = mousePos.x - position.x;
        const dy = mousePos.y - position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > CAT_SIZE) {
          isMovingRef.current = true;
          const angle = Math.atan2(dy, dx);
          const newX = position.x + Math.cos(angle) * SPEED;
          const newY = position.y + Math.sin(angle) * SPEED;

          if (idleState !== 0) {
            setIdleState(0);
            setStateCount(0);
          }

          const degrees = (angle * 180) / Math.PI;
          let newSprite = 'right';
          
          if (degrees > -22.5 && degrees <= 22.5) newSprite = 'right';
          else if (degrees > 22.5 && degrees <= 67.5) newSprite = 'downright';
          else if (degrees > 67.5 && degrees <= 112.5) newSprite = 'down';
          else if (degrees > 112.5 && degrees <= 157.5) newSprite = 'downleft';
          else if (degrees > 157.5 || degrees <= -157.5) newSprite = 'left';
          else if (degrees > -157.5 && degrees <= -112.5) newSprite = 'upleft';
          else if (degrees > -112.5 && degrees <= -67.5) newSprite = 'up';
          else if (degrees > -67.5 && degrees <= -22.5) newSprite = 'upright';

          setSprite(newSprite);
          setPosition({ x: newX, y: newY });
        } else {
          if (isMovingRef.current) {
            setIdleState(0);
            setStateCount(0);
            isMovingRef.current = false;
          }
          handleIdle();
        }
      } else {
        setSprite('sleep');
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [mousePos, position, isWaiting, mounted, idleState, stateCount]);

  if (!mounted) return null;

  const currentSpriteName = getSpriteName(sprite, frame);

  return (
    <div 
      className="fixed pointer-events-none"
      style={{
        left: `${position.x - CAT_SIZE/2}px`,
        top: `${position.y - CAT_SIZE/2}px`,
        width: `${CAT_SIZE}px`,
        height: `${CAT_SIZE}px`,
        zIndex: 9999,
      }}
    >
      <img
        src={`/assets/neko/${currentSpriteName}.png`}
        alt="Neko"
        className="w-full h-full"
      />
    </div>
  );
};

export default NekoComponent;
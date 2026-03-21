#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <algorithm>
#include <cmath>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if (keyboard)
        {
            keyboard->update();
            apply_keyboard_joystick();
        }
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;

private:
    void apply_keyboard_joystick()
    {
        auto& joy = lowstate->joystick;
        const std::string key = keyboard->key();

        // Use direct axis response for keyboard teleop.
        joy.lx.smooth = 1.0f;
        joy.ly.smooth = 1.0f;
        joy.rx.smooth = 1.0f;
        joy.LT.smooth = 1.0f;

        // Reset one-shot transition buttons every cycle.
        joy.A(0);
        joy.B(0);
        joy.X(0);
        joy.start(0);
        joy.LT(0.0f);

        // Incremental command mode:
        // WASD adjusts linear x/y, QE adjusts yaw, and command decays when no key is pressed.
        static float cmd_x = 0.0f;
        static float cmd_y = 0.0f;
        static float cmd_yaw = 0.0f;

        constexpr float kStepLin = 0.15f;
        constexpr float kStepLat = 0.15f;
        constexpr float kStepYaw = 0.15f;
        constexpr float kDecay = 0.90f;
        constexpr float kDeadband = 0.02f;

        if (key == "w")
        {
            cmd_x += kStepLin;
        }
        else if (key == "s")
        {
            cmd_x -= kStepLin;
        }
        else if (key == "a")
        {
            cmd_y += kStepLat;
        }
        else if (key == "d")
        {
            cmd_y -= kStepLat;
        }
        else if (key == "q")
        {
            cmd_yaw += kStepYaw;
        }
        else if (key == "e")
        {
            cmd_yaw -= kStepYaw;
        }
        else
        {
            cmd_x *= kDecay;
            cmd_y *= kDecay;
            cmd_yaw *= kDecay;
        }

        cmd_x = std::clamp(cmd_x, -1.0f, 1.0f);
        cmd_y = std::clamp(cmd_y, -1.0f, 1.0f);
        cmd_yaw = std::clamp(cmd_yaw, -1.0f, 1.0f);

        if (std::fabs(cmd_x) < kDeadband) cmd_x = 0.0f;
        if (std::fabs(cmd_y) < kDeadband) cmd_y = 0.0f;
        if (std::fabs(cmd_yaw) < kDeadband) cmd_yaw = 0.0f;

        // Match observation mapping in velocity_commands:
        // cmd_x <- ly, cmd_y <- -lx, cmd_yaw <- -rx
        joy.ly(cmd_x);
        joy.lx(-cmd_y);
        joy.rx(-cmd_yaw);

        // Keyboard FSM transitions (no physical remote needed):
        // 1: A (FixStand), 2: B (Passive), 3: X (Velocity/RL)
        if (keyboard->on_pressed)
        {
            if (key == "1")
            {
                joy.A(1);
            }
            else if (key == "2")
            {
                joy.B(1);
            }
            else if (key == "3")
            {
                joy.X(1);
            }
            else if (key == "0" || key == "p")
            {
                joy.B(1);
            }
            else if (key == " ")
            {
                cmd_x = 0.0f;
                cmd_y = 0.0f;
                cmd_yaw = 0.0f;
                joy.ly(0.0f);
                joy.lx(0.0f);
                joy.rx(0.0f);
            }
        }
    }
};
